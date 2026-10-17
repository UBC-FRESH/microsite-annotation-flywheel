"""Read from and write to Roboflow.

This module is the only place that talks to Roboflow. It exposes small records
(:class:`ProjectInfo`, :class:`ImageSummary`, :class:`ImageDetail`) and a client whose
network seams are injectable, so the rest of the flywheel can be tested without a network.

Two facts drive the design, both verified live and recorded in
``notes/reference/roboflow-canonical-projects.md``:

- An image's annotation status comes from the image listing; ``annotations`` is empty for an
  annotation-free image and carries a ``count`` otherwise. This is the guard the uploader
  checks before every write.
- ``save_annotation(..., is_prediction=True, annotation_overwrite=False)`` attaches a
  prediction to an existing image and refuses to replace an existing annotation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from geometry import DetectedObject, objects_from_annotation

API_BASE: str = "https://api.roboflow.com"
SEARCH_PAGE: int = 100


class HttpSession(Protocol):
    """The slice of ``requests.Session`` this module uses, so tests can inject a fake."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Perform an HTTP GET and return a response-like object.

        Parameters
        ----------
        url : str
            Request URL.
        params : mapping of str to str or None, optional
            Query parameters.
        timeout : float or None, optional
            Request timeout in seconds.

        Returns
        -------
        Any
            A response-like object.
        """
        ...


class RoboflowError(RuntimeError):
    """Raised when a Roboflow call fails or returns an unexpected payload."""


@dataclass(frozen=True)
class ProjectInfo:
    """A Roboflow project's identity and annotation type.

    Attributes
    ----------
    slug : str
        Project slug within its workspace.
    name : str
        Human-readable project name.
    project_type : str
        Roboflow project type, such as ``instance-segmentation``.
    annotation : str
        The project's annotation group name.
    """

    slug: str
    name: str
    project_type: str
    annotation: str


@dataclass(frozen=True)
class ImageSummary:
    """One row of a project's image listing.

    Attributes
    ----------
    id : str
        Roboflow image id, used to fetch detail and to upload.
    name : str
        Image file name.
    split : str
        ``train``, ``valid``, or ``test``.
    annotation_count : int
        Number of annotation instances on the image; ``0`` means annotation-free.
    """

    id: str
    name: str
    split: str
    annotation_count: int

    @property
    def is_annotated(self) -> bool:
        """Return whether the image already carries any annotation."""
        return self.annotation_count > 0


@dataclass(frozen=True)
class ImageDetail:
    """An image's geometry, source URL, and current objects.

    Attributes
    ----------
    id : str
        Roboflow image id.
    name : str
        Image file name.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    objects : list of DetectedObject
        The objects currently annotated on the image.
    image_url : str
        URL of the source image.
    """

    id: str
    name: str
    width: int
    height: int
    objects: list[DetectedObject]
    image_url: str


# The roboflow SDK is dynamically typed, so its project object is passed around as Any with
# this alias; every call site narrows or casts what it needs.
ProjectLoader = Callable[[str, str, str], Any]


def _default_project_loader(api_key: str, workspace: str, slug: str) -> Any:
    """Build a Roboflow SDK project object.

    Parameters
    ----------
    api_key : str
        Roboflow API key.
    workspace : str
        Workspace slug.
    slug : str
        Project slug.

    Returns
    -------
    Any
        A ``roboflow`` SDK project object. Imported lazily so the package stays light.

    Raises
    ------
    RoboflowError
        When the SDK cannot be imported or the project cannot be loaded.
    """
    try:
        import roboflow  # imported lazily to keep the import graph light
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK installed
        raise RoboflowError("the 'roboflow' package is required but not installed") from exc
    try:
        client = roboflow.Roboflow(api_key=api_key)
        return client.workspace(workspace).project(slug)
    except Exception as exc:  # the SDK raises a variety of error types
        raise RoboflowError(f"could not load project '{workspace}/{slug}': {exc}") from exc


class RoboflowClient:
    """Read and write Roboflow project data.

    Parameters
    ----------
    api_key : str
        Roboflow API key.
    workspace : str
        Workspace slug.
    project_loader : ProjectLoader or None, optional
        Builds an SDK project object. Injected in tests to avoid the network. Defaults to
        the real SDK loader.
    session : HttpSession or None, optional
        HTTP session for REST calls. Injected in tests. Defaults to a new requests session.

    Attributes
    ----------
    _session : HttpSession
        The HTTP session used for REST calls.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        *,
        project_loader: ProjectLoader | None = None,
        session: HttpSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._workspace = workspace
        self._project_loader = project_loader or _default_project_loader
        self._session = session or requests.Session()
        self._projects: dict[str, Any] = {}

    def _project(self, slug: str) -> Any:
        """Return a cached SDK project object.

        Parameters
        ----------
        slug : str
            Project slug.

        Returns
        -------
        Any
            The SDK project object.
        """
        if slug not in self._projects:
            self._projects[slug] = self._project_loader(self._api_key, self._workspace, slug)
        return self._projects[slug]

    def project_info(self, slug: str) -> ProjectInfo:
        """Fetch a project's identity and annotation type over REST.

        Parameters
        ----------
        slug : str
            Project slug.

        Returns
        -------
        ProjectInfo
            The project record.

        Raises
        ------
        RoboflowError
            When the request fails or the payload lacks a ``project`` object.
        """
        payload = self._get_json(f"/{self._workspace}/{slug}")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise RoboflowError(f"project '{slug}' returned no project object")
        return ProjectInfo(
            slug=slug,
            name=str(project.get("name", slug)),
            project_type=str(project.get("type", "")),
            annotation=str(project.get("annotation", "")),
        )

    def iter_images(self, slug: str) -> Iterator[ImageSummary]:
        """Iterate every image in a project, newest listing order.

        Parameters
        ----------
        slug : str
            Project slug.

        Yields
        ------
        ImageSummary
            One record per image. Annotation-free images have ``annotation_count`` zero.

        Raises
        ------
        RoboflowError
            When the SDK search fails or returns an unexpected payload.
        """
        project = self._project(slug)
        offset = 0
        while True:
            try:
                page = project.search(
                    limit=SEARCH_PAGE,
                    offset=offset,
                    fields=["id", "name", "annotations", "split"],
                )
            except Exception as exc:  # the SDK raises a variety of error types
                raise RoboflowError(f"listing images for '{slug}' failed: {exc}") from exc
            if not isinstance(page, list):
                raise RoboflowError(f"image listing for '{slug}' was not a list")
            for item in page:
                yield _summary_from(item)
            if len(page) < SEARCH_PAGE:
                break
            offset += SEARCH_PAGE

    def image_detail(self, slug: str, image_id: str) -> ImageDetail:
        """Fetch one image's geometry, objects, and source URL.

        Parameters
        ----------
        slug : str
            Project slug.
        image_id : str
            Roboflow image id.

        Returns
        -------
        ImageDetail
            The image detail record.

        Raises
        ------
        RoboflowError
            When the request fails or the payload lacks an ``image`` object.
        """
        payload = self._get_json(f"/{self._workspace}/{slug}/images/{image_id}")
        image = payload.get("image")
        if not isinstance(image, dict):
            raise RoboflowError(f"image '{image_id}' returned no image object")
        annotation = image.get("annotation") or {}
        urls = image.get("urls") or {}
        return ImageDetail(
            id=str(image.get("id", image_id)),
            name=str(image.get("name", image_id)),
            width=int(annotation.get("width") or 0),
            height=int(annotation.get("height") or 0),
            objects=objects_from_annotation(annotation),
            image_url=str(urls.get("original", "")),
        )

    def download(self, url: str, destination: Path) -> int:
        """Download a URL to a file.

        Parameters
        ----------
        url : str
            Source URL.
        destination : Path
            Destination file; its parent directory is created.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        RoboflowError
            When the download fails or returns a non-success status.
        """
        if not url:
            raise RoboflowError("cannot download an empty URL")
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self._session.get(url, timeout=120)
        if response.status_code != 200:
            raise RoboflowError(f"download of {url} failed with HTTP {response.status_code}")
        destination.write_bytes(response.content)
        return len(response.content)

    def upload_annotation(
        self,
        slug: str,
        image_id: str,
        document: Mapping[str, Any],
        *,
        name: str = "annotation.coco.json",
        labelmap: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """Attach a prediction annotation to an existing image.

        Never overwrites an existing annotation: ``annotation_overwrite`` is fixed ``False``
        and the annotation is marked ``is_prediction``.

        Parameters
        ----------
        slug : str
            Project slug.
        image_id : str
            Roboflow image id to attach the annotation to.
        document : mapping of str to Any
            The annotation document, serialised to JSON (COCO by default).
        name : str, optional
            File name that tells Roboflow the annotation format.
        labelmap : mapping of str to str or None, optional
            Numeric class id to class name, for formats that need it.

        Returns
        -------
        mapping of str to Any
            The server response.

        Raises
        ------
        RoboflowError
            When the upload fails.
        """
        project = self._project(slug)
        annotation_path = {"rawText": json.dumps(document), "name": name}
        try:
            result = project.save_annotation(
                annotation_path=annotation_path,
                annotation_labelmap=dict(labelmap) if labelmap else None,
                image_id=image_id,
                is_prediction=True,
                annotation_overwrite=False,
            )
        except Exception as exc:  # the SDK raises a variety of error types
            raise RoboflowError(
                f"uploading annotation to image '{image_id}' failed: {exc}"
            ) from exc
        if isinstance(result, tuple):
            return {"response": result[0]}
        return {"response": result}

    def _get_json(self, path: str, params: Mapping[str, str] | None = None) -> dict[str, Any]:
        """GET a REST path and decode the JSON body.

        Parameters
        ----------
        path : str
            Path beginning with ``/``.
        params : mapping of str to str or None, optional
            Extra query parameters. The API key is always added.

        Returns
        -------
        dict of str to Any
            The decoded JSON object.

        Raises
        ------
        RoboflowError
            When the request fails, the status is not 200, or the body is not an object.
        """
        query = {"api_key": self._api_key}
        if params:
            query.update(params)
        response = self._session.get(f"{API_BASE}{path}", params=query, timeout=60)
        if response.status_code != 200:
            raise RoboflowError(f"GET {path} failed with HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RoboflowError(f"GET {path} returned a non-object body")
        return payload


def _summary_from(item: Any) -> ImageSummary:
    """Build an :class:`ImageSummary` from one SDK search record.

    Parameters
    ----------
    item : Any
        One record from ``project.search``.

    Returns
    -------
    ImageSummary
        The parsed summary.
    """
    annotations = item.get("annotations")
    if isinstance(annotations, dict):
        count = int(annotations.get("count") or 0)
    elif isinstance(annotations, list):
        count = len(annotations)
    else:
        count = 0
    return ImageSummary(
        id=str(item.get("id", "")),
        name=str(item.get("name", "")),
        split=str(item.get("split", "")),
        annotation_count=count,
    )

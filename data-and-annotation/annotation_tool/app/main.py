from __future__ import annotations

import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.requests import Request

from .agreement import compute_agreement_summary, write_agreement_summary_file
from .config import (
    APP_ROOT,
    AUDIO_BASE_DIRS,
    CREDENTIALS_PATH,
    REPO_ROOT,
    RESULTS_ROOT,
    SUPPORTED_CATEGORIES,
)
from .manifest_loader import ManifestItem, load_manifest, manifest_by_item_id
from .results_store import (
    append_result_row,
    available_result_paths,
    completed_item_ids,
    load_result_rows,
    load_result_rows_for_categories,
    reset_results_for_annotator,
    validate_annotator_id,
)
from .session_store import clear_session, create_session, get_session
from .summary import compute_summary
from .user_store import UserRecord, authenticate_user, load_users

app = FastAPI(title="VocalGrad Manual Annotation Tool")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_user_record(user: UserRecord) -> dict[str, Any]:
    return {
        "user_id": user.user_id,
        "role": user.role,
        "assigned_categories": list(user.assigned_categories),
        "category_order": list(user.category_order),
    }


def manifest_items(category: str) -> tuple[ManifestItem, ...]:
    return load_manifest(category)


def manifest_index(category: str) -> dict[str, ManifestItem]:
    return manifest_by_item_id(category)


def require_session(authorization: str | None, *, roles: set[str] | None = None) -> UserRecord:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization token.")
    token = authorization.removeprefix("Bearer ").strip()
    session = get_session(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    user = load_users().get(session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    if roles is not None and user.role not in roles:
        raise HTTPException(status_code=403, detail="You do not have access to this resource.")
    return user


def category_progress(user: UserRecord) -> list[dict[str, Any]]:
    progress: list[dict[str, Any]] = []
    for index, category in enumerate(user.category_order, start=1):
        items = manifest_items(category)
        completed = len(completed_item_ids(category, user.user_id))
        total = len(items)
        progress.append(
            {
                "category": category,
                "position": index,
                "completed_items": completed,
                "total_items": total,
                "is_complete": completed >= total,
            }
        )
    return progress


def first_remaining_item(user: UserRecord) -> tuple[str | None, ManifestItem | None]:
    for category in user.category_order:
        done = completed_item_ids(category, user.user_id)
        for item in manifest_items(category):
            if item.item_id not in done:
                return category, item
    return None, None


def item_to_public_payload(
    item: ManifestItem,
    *,
    category: str,
    category_position: int,
    category_total: int,
    category_completed: int,
    overall_completed: int,
    overall_total: int,
) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "clip_id": item.clip_id,
        "audio_url": item.audio_url(),
        "attribute": item.attribute,
        "category": category,
        "progress": {
            "category_position": category_position,
            "category_count": category_total,
            "completed_in_category": category_completed,
            "total_in_category": len(manifest_items(category)),
            "completed_overall": overall_completed,
            "total_overall": overall_total,
        },
    }


def build_annotator_state(user: UserRecord) -> dict[str, Any]:
    progress = category_progress(user)
    overall_completed = sum(row["completed_items"] for row in progress)
    overall_total = sum(row["total_items"] for row in progress)
    current_category, current_item = first_remaining_item(user)

    payload: dict[str, Any] = {
        "user": public_user_record(user),
        "category_progress": progress,
        "overall_progress": {
            "completed_items": overall_completed,
            "total_items": overall_total,
            "completed_categories": sum(1 for row in progress if row["is_complete"]),
            "total_categories": len(progress),
        },
        "is_complete": current_item is None,
        "current_item": None,
    }
    if current_category is not None and current_item is not None:
        category_row = next(row for row in progress if row["category"] == current_category)
        payload["current_item"] = item_to_public_payload(
            current_item,
            category=current_category,
            category_position=category_row["position"],
            category_total=len(progress),
            category_completed=category_row["completed_items"],
            overall_completed=overall_completed,
            overall_total=overall_total,
        )
    return payload


def summary_for_user(user: UserRecord) -> dict[str, Any]:
    rows = load_result_rows_for_categories(user.user_id, user.category_order)
    return compute_summary(rows, annotator_id=user.user_id)


def admin_dashboard_payload() -> dict[str, Any]:
    users = load_users()
    annotators = [user for user in users.values() if user.role == "annotator"]
    test_users = [user for user in users.values() if user.role == "test"]

    overall_by_category: dict[str, Any] = {}
    agreement_by_category: dict[str, Any] = {}
    for category in SUPPORTED_CATEGORIES:
        category_rows: list[dict[str, Any]] = []
        for user in annotators:
            if category in user.category_order:
                category_rows.extend(load_result_rows(category, user.user_id))
        overall_by_category[category] = compute_summary(category_rows)
        agreement_by_category[category] = compute_agreement_summary(category)

    annotator_summaries = []
    for user in annotators:
        per_category = {
            category: compute_summary(
                load_result_rows(category, user.user_id),
                annotator_id=user.user_id,
            )
            for category in user.category_order
        }
        annotator_summaries.append(
            {
                "user": public_user_record(user),
                "overall_summary": summary_for_user(user),
                "per_category_summary": per_category,
            }
        )

    test_user_summaries = []
    for user in test_users:
        per_category = {
            category: compute_summary(
                load_result_rows(category, user.user_id),
                annotator_id=user.user_id,
            )
            for category in user.category_order
        }
        test_user_summaries.append(
            {
                "user": public_user_record(user),
                "overall_summary": summary_for_user(user),
                "per_category_summary": per_category,
            }
        )

    raw_downloads = []
    for category in SUPPORTED_CATEGORIES:
        for path in available_result_paths(categories=[category]):
            raw_downloads.append(
                {
                    "category": category,
                    "annotator_id": path.stem,
                    "filename": path.name,
                    "download_url": (
                        f"/api/admin/download/raw?category={category}&annotator_id={path.stem}"
                    ),
                }
            )

    return {
        "generated_at": now_utc(),
        "credentials_path": str(CREDENTIALS_PATH),
        "bulk_raw_download_url": "/api/admin/download/raw-all",
        "overall_by_category": overall_by_category,
        "agreement_by_category": agreement_by_category,
        "annotators": annotator_summaries,
        "test_users": test_user_summaries,
        "resettable_users": [
            public_user_record(user)
            for user in users.values()
            if user.role in {"annotator", "test"}
        ],
        "raw_downloads": raw_downloads,
    }


def build_raw_results_archive() -> Path:
    raw_paths = available_result_paths()
    if not raw_paths:
        raise HTTPException(status_code=404, detail="No raw annotation files found.")

    with tempfile.NamedTemporaryFile(
        prefix="vocalgrad-raw-annotations-",
        suffix=".zip",
        delete=False,
    ) as temp_file:
        archive_path = Path(temp_file.name)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in raw_paths:
            archive.write(path, arcname=str(path.relative_to(RESULTS_ROOT)))

    return archive_path


class LoginInput(BaseModel):
    user_id: str
    password: str


class AnnotationSubmitInput(BaseModel):
    category: str
    item_id: str
    annotated_label: str
    response_time_ms: int
    replay_count: int
    started_at: str
    submitted_at: str


class ResetUserInput(BaseModel):
    user_id: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "supported_categories": SUPPORTED_CATEGORIES,
            "manifest_name": "shared_annotation_50.csv",
        },
    )


@app.post("/api/auth/login")
def login(payload: LoginInput) -> dict[str, Any]:
    user = authenticate_user(payload.user_id, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid user ID or password.")

    session = create_session(user_id=user.user_id, role=user.role)
    response: dict[str, Any] = {
        "session_token": session.token,
        "user": public_user_record(user),
    }
    if user.role == "admin":
        response["dashboard"] = admin_dashboard_payload()
    else:
        response["state"] = build_annotator_state(user)
        if response["state"]["is_complete"]:
            response["summary"] = summary_for_user(user)
    return response


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    if authorization and authorization.startswith("Bearer "):
        clear_session(authorization.removeprefix("Bearer ").strip())
    return {"logged_out": True}


@app.get("/api/session/state")
def get_session_state(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = require_session(authorization)
    if user.role == "admin":
        return {
            "user": public_user_record(user),
            "dashboard": admin_dashboard_payload(),
        }

    payload = {
        "user": public_user_record(user),
        "state": build_annotator_state(user),
    }
    if payload["state"]["is_complete"]:
        payload["summary"] = summary_for_user(user)
    return payload


@app.post("/api/annotation/submit")
def submit_annotation(
    payload: AnnotationSubmitInput,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    user = require_session(authorization, roles={"annotator", "test"})
    if payload.category not in user.category_order:
        raise HTTPException(status_code=403, detail="Category is not assigned to this user.")

    annotated_label = payload.annotated_label.strip().lower()
    if annotated_label not in {"increase", "decrease"}:
        raise HTTPException(status_code=400, detail="annotated_label must be increase or decrease.")
    if payload.response_time_ms < 0:
        raise HTTPException(status_code=400, detail="response_time_ms must be non-negative.")
    if payload.replay_count < 0:
        raise HTTPException(status_code=400, detail="replay_count must be non-negative.")

    if payload.item_id in completed_item_ids(payload.category, user.user_id):
        raise HTTPException(status_code=409, detail="This item has already been submitted.")

    item = manifest_index(payload.category).get(payload.item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item_id not found in manifest.")

    row = {
        "annotation_id": str(uuid.uuid4()),
        "annotator_id": user.user_id,
        "item_id": item.item_id,
        "clip_id": item.clip_id,
        "category": item.category,
        "attribute": item.attribute,
        "difficulty": item.difficulty,
        "trajectory": item.trajectory,
        "audio_path": item.audio_path,
        "ground_truth_label": item.label,
        "annotated_label": annotated_label,
        "is_correct": annotated_label == item.label,
        "response_time_ms": payload.response_time_ms,
        "replay_count": payload.replay_count,
        "started_at": payload.started_at,
        "submitted_at": payload.submitted_at,
    }
    append_result_row(payload.category, user.user_id, row)
    write_agreement_summary_file(payload.category)

    state = build_annotator_state(user)
    response = {
        "saved": True,
        "state": state,
    }
    if state["is_complete"]:
        response["summary"] = summary_for_user(user)
    return response


@app.get("/api/annotator/summary")
def get_annotator_summary(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = require_session(authorization, roles={"annotator", "test"})
    return summary_for_user(user)


@app.get("/api/admin/dashboard")
def get_admin_dashboard(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_session(authorization, roles={"admin"})
    return admin_dashboard_payload()


@app.get("/api/admin/download/raw")
def download_raw_results(
    category: str,
    annotator_id: str,
    authorization: str | None = Header(default=None),
) -> FileResponse:
    require_session(authorization, roles={"admin"})
    validate_annotator_id(annotator_id)
    path = Path(
        str(
            next(
                (
                    p
                    for p in available_result_paths(
                        categories=[category],
                        annotator_id=annotator_id,
                    )
                    if p.stem == annotator_id
                ),
                "",
            )
        )
    )
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path)


@app.get("/api/admin/download/raw-all")
def download_all_raw_results(authorization: str | None = Header(default=None)) -> FileResponse:
    require_session(authorization, roles={"admin"})
    archive_path = build_raw_results_archive()
    return FileResponse(
        archive_path,
        filename="vocalgrad_raw_annotation_files.zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@app.post("/api/admin/reset-test-user")
def reset_test_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_session(authorization, roles={"admin"})
    users = load_users()
    test_users = [user for user in users.values() if user.role == "test"]
    if len(test_users) != 1:
        raise HTTPException(status_code=400, detail="Expected exactly one test user.")

    test_user = test_users[0]
    removed = reset_results_for_annotator(test_user.user_id, test_user.category_order)
    return {
        "reset_user_id": test_user.user_id,
        "removed_paths": [str(path) for path in removed],
        "dashboard": admin_dashboard_payload(),
    }


@app.post("/api/admin/reset-user")
def reset_user(
    payload: ResetUserInput,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_session(authorization, roles={"admin"})
    users = load_users()
    user = users.get(payload.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.role == "admin":
        raise HTTPException(status_code=400, detail="Admin data cannot be reset.")

    removed = reset_results_for_annotator(user.user_id, user.category_order)
    return {
        "reset_user_id": user.user_id,
        "removed_paths": [str(path) for path in removed],
        "dashboard": admin_dashboard_payload(),
    }


@app.get("/audio/{path:path}")
def serve_audio(path: str) -> FileResponse:
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    allowed = any(
        base.resolve() in resolved.parents or resolved == base.resolve()
        for base in AUDIO_BASE_DIRS
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="Audio path is outside allowed directories.")

    return FileResponse(resolved)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "supported_categories": list(SUPPORTED_CATEGORIES),
        "server_time": now_utc(),
        "users_file_exists": CREDENTIALS_PATH.parent.joinpath("users.json").exists(),
    }

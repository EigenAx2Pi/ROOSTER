"""
ROOSTER Web Application - FastAPI Backend
"""

import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.rooster import (
    clean_roster_excel,
    get_working_days,
    generate_booking_predictions,
    apply_predictions_to_template,
    generate_predictions_to_excel,
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ROOSTER - Rule-Based Employee Roster Prediction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_upload(upload_file: UploadFile, prefix: str) -> Path:
    """Save an uploaded file with a unique name and return its path."""
    ext = Path(upload_file.filename).suffix
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return file_path


@app.post("/api/predict")
async def predict(
    roster_file: UploadFile = File(..., description="Historical roster Excel file"),
    holiday_file: UploadFile = File(..., description="Holiday calendar Excel file"),
    template_file: UploadFile = File(None, description="Optional blank template Excel file"),
    month: int = Form(..., description="Target month (1-12)"),
    year: int = Form(..., description="Target year"),
    threshold: float = Form(0.6, description="Booking frequency threshold (0-1)"),
    min_days_per_week: int = Form(3, description="Minimum bookings per week"),
):
    """
    Run the ROOSTER prediction pipeline.
    Accepts roster file, holiday file, optional template, and parameters.
    Returns a download link for the generated predictions Excel file.
    """
    if not 1 <= month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    if not 2000 <= year <= 2100:
        raise HTTPException(status_code=400, detail="Year must be between 2000 and 2100.")
    if not 0 < threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")
    if not 1 <= min_days_per_week <= 5:
        raise HTTPException(status_code=400, detail="Min days per week must be between 1 and 5.")

    try:
        roster_path = save_upload(roster_file, "roster")
        holiday_path = save_upload(holiday_file, "holiday")
        template_path = (
            save_upload(template_file, "template")
            if template_file and template_file.filename
            else None
        )

        # Step 1: Clean historical roster
        df_cleaned = clean_roster_excel(str(roster_path))

        # Step 2: Generate working days
        working_days = get_working_days(month, year, str(holiday_path))

        # Step 3: Generate predictions
        predicted_df = generate_booking_predictions(
            df_cleaned, working_days,
            min_days_per_week=min_days_per_week,
            threshold=threshold,
        )

        # Step 4: Output to Excel
        output_filename = f"Roster_Prediction_{year}_{month:02d}_{uuid.uuid4().hex[:8]}.xlsx"
        output_path = OUTPUT_DIR / output_filename

        if template_path:
            apply_predictions_to_template(predicted_df, str(template_path), str(output_path))
        else:
            generate_predictions_to_excel(predicted_df, str(output_path))

        # Cleanup uploaded files
        for p in [roster_path, holiday_path, template_path]:
            if p and p.exists():
                p.unlink()

        return JSONResponse({
            "status": "success",
            "message": "Predictions generated successfully.",
            "download_url": f"/api/download/{output_filename}",
            "filename": output_filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/download/{filename}")
async def download(filename: str):
    """Download a generated prediction file."""
    file_path = (OUTPUT_DIR / filename).resolve()
    if not file_path.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Serve static frontend files
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

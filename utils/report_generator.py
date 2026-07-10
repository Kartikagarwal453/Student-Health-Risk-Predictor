"""PDF report generation for individual student predictions."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from utils.constants import APP_TITLE


def generate_prediction_pdf(
    student: dict[str, Any],
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
    lifestyle_score: int,
    recommendations: list[dict[str, str]],
) -> bytes:
    """Generate a polished prediction report as PDF bytes."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError(
            "PDF generation requires reportlab. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles["Title"].textColor = colors.HexColor("#0f172a")

    story = [
        Paragraph(APP_TITLE, styles["Title"]),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Prediction Summary", styles["SectionHeading"]),
    ]

    summary_rows = [
        ["Prediction", prediction.upper()],
        ["Confidence", f"{confidence * 100:.2f}%"],
        ["Lifestyle Score", f"{lifestyle_score}/100"],
    ]
    story.append(_styled_table(summary_rows))

    story.append(Paragraph("Prediction Probability", styles["SectionHeading"]))
    probability_rows = [
        [class_name, f"{probability * 100:.2f}%"]
        for class_name, probability in probabilities.items()
    ]
    story.append(_styled_table(probability_rows, header=["Class", "Probability"]))

    story.append(Paragraph("Student Information", styles["SectionHeading"]))
    info_rows = [[key.replace("_", " ").title(), str(value)] for key, value in student.items()]
    story.append(_styled_table(info_rows))

    story.append(Paragraph("Recommendations", styles["SectionHeading"]))
    recommendation_rows = [
        [
            item["priority"],
            item["area"],
            Paragraph(item["recommendation"], styles["Normal"]),
        ]
        for item in recommendations
    ]
    story.append(_styled_table(recommendation_rows, header=["Priority", "Area", "Recommendation"]))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _styled_table(rows: list[list[Any]], header: list[str] | None = None) -> Table:
    """Create a consistently styled ReportLab table."""

    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table_data = [header] + rows if header else rows
    table = Table(table_data, hAlign="LEFT", repeatRows=1 if header else 0)
    style_commands = [
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if header:
        style_commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(style_commands))
    return table

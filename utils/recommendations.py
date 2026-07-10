"""Personalized recommendation engine for student health predictions."""

from __future__ import annotations

from typing import Any


def generate_recommendations(student: dict[str, Any]) -> list[dict[str, str]]:
    """Generate prioritized recommendations from student inputs."""

    recommendations: list[dict[str, str]] = []

    if float(student["sleep_duration"]) < 7:
        recommendations.append(
            {
                "priority": "High",
                "area": "Sleep",
                "recommendation": "Increase sleep toward a consistent 7 to 9 hour routine.",
            }
        )
    if student["sleep_quality"] == "poor":
        recommendations.append(
            {
                "priority": "High",
                "area": "Sleep quality",
                "recommendation": "Reduce screen exposure before bed and keep a fixed sleep schedule.",
            }
        )
    if student["stress_level"] == "high":
        recommendations.append(
            {
                "priority": "High",
                "area": "Stress",
                "recommendation": "Add a daily stress reset such as breathing practice, journaling, or a walk.",
            }
        )
    if float(student["exercise_duration"]) < 30:
        recommendations.append(
            {
                "priority": "Medium",
                "area": "Exercise",
                "recommendation": "Aim for at least 30 minutes of moderate activity on most days.",
            }
        )
    if float(student["step_count"]) < 7000:
        recommendations.append(
            {
                "priority": "Medium",
                "area": "Activity",
                "recommendation": "Build daily movement gradually, starting with an extra 1000 steps.",
            }
        )
    if float(student["water_intake"]) < 2:
        recommendations.append(
            {
                "priority": "Low",
                "area": "Hydration",
                "recommendation": "Keep water visible during study hours and target at least 2 liters daily.",
            }
        )
    bmi = float(student["bmi"])
    if bmi < 18.5:
        recommendations.append(
            {
                "priority": "Medium",
                "area": "BMI",
                "recommendation": "Consider nutrition counseling to support healthy weight gain.",
            }
        )
    elif bmi >= 30:
        recommendations.append(
            {
                "priority": "High",
                "area": "BMI",
                "recommendation": "Consult a qualified clinician for a sustainable weight management plan.",
            }
        )
    if student["smoking_alcohol"] in {"yes", "occasional"}:
        recommendations.append(
            {
                "priority": "High",
                "area": "Substance use",
                "recommendation": "Reducing smoking or alcohol exposure can materially improve health risk.",
            }
        )
    if student["physical_activity_level"] == "sedentary":
        recommendations.append(
            {
                "priority": "Medium",
                "area": "Activity level",
                "recommendation": "Break long sitting periods with short movement breaks every hour.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "Maintain",
                "area": "Lifestyle",
                "recommendation": "Current habits look strong. Keep monitoring sleep, stress, and hydration.",
            }
        )

    return recommendations

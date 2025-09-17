# Cell 1: Install dependencies and imports

import json
import numpy as np
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import psycopg2
from psycopg2.extras import Json

print("All imports successful!")

# Cell 2 FIXED: Enums and Data Classes with proper JSON serialization

class DisasterType(Enum):
    TSUNAMI = "tsunami"
    HURRICANE = "hurricane"
    STORM_SURGE = "storm_surge"
    COASTAL_FLOODING = "coastal_flooding"
    RIP_CURRENT = "rip_current"
    HIGH_WAVES = "high_waves"
    ALGAE_BLOOM = "algae_bloom"
    OIL_SPILL = "oil_spill"
    EROSION = "coastal_erosion"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    WARNING = 1
    EMERGENCY = 2
    EVACUATION = 3


@dataclass
class DisasterReport:
    disaster_type: DisasterType
    confidence_score: float
    reliability_score: float
    alert_level: AlertLevel
    location: Dict[str, float]
    timestamp: datetime
    description: str
    image_analysis: Dict
    schema: Dict
    report_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization of enums and datetime"""
        return {
            "disaster_type": self.disaster_type.value if self.disaster_type else "unknown",
            "confidence_score": self.confidence_score,
            "reliability_score": self.reliability_score,
            "alert_level": self.alert_level.value if self.alert_level else 1,
            "alert_level_name": self.alert_level.name if self.alert_level else "WARNING",
            "location": self.location or {"lat": 0.0, "lon": 0.0},
            "timestamp": self.timestamp.isoformat() if self.timestamp else datetime.now().isoformat(),
            "description": self.description or "",
            "image_analysis": self.image_analysis or {},
            "schema": self.schema or {},
            "report_id": self.report_id
        }

# Helper function to prevent NoneType errors
def safe_len(obj):
    """Safe length function that handles None values"""
    return len(obj) if obj is not None else 0

# Custom JSON encoder for handling remaining enum serialization issues
class DisasterJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle DisasterType and AlertLevel enums"""
    def default(self, obj):
        if isinstance(obj, DisasterType):
            return obj.value
        elif isinstance(obj, AlertLevel):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

print("Fixed enums and data classes with proper JSON serialization defined successfully!")

# Cell 3: OceanDisasterDetector Class - Initialization and Setup Methods

class OceanDisasterDetector:
    def __init__(self, output_path: str = "disaster_reports.json"):
        """
        Initialize the detector with user-provided database.
        """
        self.output_path = output_path
        self.disaster_keywords = self._initialize_disaster_keywords()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.db_connection = psycopg2.connect(
            dbname="bluebeacon",
            user="jeevan",
            password="Jeevan@1210",
            host="15.207.102.171",
            port=5432,
        )

    def _initialize_disaster_keywords(self) -> Dict[DisasterType, List[str]]:
        return {
            DisasterType.TSUNAMI: [
                "tsunami", "tidal wave", "seismic sea wave", "massive wave",
                "water rushing inland", "sea level rise", "earthquake wave",
            ],
            DisasterType.HURRICANE: [
                "hurricane", "cyclone", "typhoon", "storm", "wind damage",
                "rotating storm", "eye wall", "storm system",
            ],
            DisasterType.STORM_SURGE: [
                "storm surge", "surge flooding", "sea level surge",
                "coastal inundation", "storm tide", "water surge",
            ],
            DisasterType.COASTAL_FLOODING: [
                "coastal flooding", "sea flooding", "tidal flooding",
                "water intrusion", "flood water", "inundation",
            ],
            DisasterType.RIP_CURRENT: [
                "rip current", "dangerous current", "strong undertow",
                "water pulling", "current danger", "swimming hazard",
            ],
            DisasterType.HIGH_WAVES: [
                "high waves", "large waves", "dangerous waves",
                "wave height", "surf danger", "wave action",
            ],
            DisasterType.ALGAE_BLOOM: [
                "algae bloom", "red tide", "harmful algae",
                "water discoloration", "toxic algae", "bloom event",
            ],
            DisasterType.OIL_SPILL: [
                "oil spill", "petroleum leak", "oil contamination",
                "oil slick", "marine pollution", "oil discharge",
            ],
            DisasterType.EROSION: [
                "coastal erosion", "beach erosion", "shoreline retreat",
                "cliff collapse", "sand loss", "erosion damage",
            ],
        }

# Cell 4 FIXED: Data Loading and Database Methods with proper JSON serialization

# Add these methods to OceanDisasterDetector class
def load_pending_reports(self) -> List[Dict[str, Any]]:
    cur = self.db_connection.cursor()
    query = """
        SELECT
            r.report_id,
            r.description,
            r.disaster_type,
            r.timestamp,
            r.user_id,
            l.latitude,
            l.longitude,
            l.location_source,
            l.address,
            l.is_coastal
        FROM reports r
        LEFT JOIN locations l ON r.report_id = l.report_id
        LEFT JOIN analysis_results ar ON ar.report_id = r.report_id
        WHERE r.status = 'pending'
          AND ar.report_id IS NULL
        ORDER BY r.created_at
    """
    cur.execute(query)
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        rid, desc, sel_dis, ts, uid, lat, lon, src, addr, is_coastal = row
        out.append({
            "report_id": rid,
            "description": desc or "",
            "selected_disaster": sel_dis,
            "timestamp": ts,
            "user_id": uid,
            "location": ({
                "lat": lat,
                "lon": lon,
                "source": src,
                "address": addr,
                "is_coastal": bool(is_coastal) if is_coastal is not None else None,
            } if lat is not None and lon is not None else None),
            "images": self._load_images_for_report(rid),
        })
    return out


def _load_images_for_report(self, report_id: str) -> List[str]:
    if not report_id:
        return []
    cur = self.db_connection.cursor()
    cur.execute(
        """
        SELECT image_data, image_path
        FROM images
        WHERE report_id = %s
        ORDER BY id
        """,
        (report_id,),
    )
    images: List[str] = []
    for image_b64, image_path in cur.fetchall():
        if image_b64:
            images.append(image_b64)
        elif image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception as e:
                print(f"Error loading image from path {image_path}: {e}")
    return images

def _update_report_status(self, report_id: Optional[str], status: str, error_msg: Optional[str] = None):
    if not report_id:
        return
    cur = self.db_connection.cursor()
    if error_msg:
        cur.execute(
            """
            UPDATE reports
               SET status = %s, error_message = %s
             WHERE report_id = %s
            """,
            (status, error_msg, report_id),
        )
    else:
        cur.execute(
            """
            UPDATE reports
               SET status = %s
             WHERE report_id = %s
            """,
            (status, report_id),
        )
    self.db_connection.commit()

def _save_analysis_result(self, report: DisasterReport):
    if not report:
        return
    cur = self.db_connection.cursor()
    payload = {
        "image_analysis": report.image_analysis or {},
        "schema": report.schema or {},
    }
    cur.execute(
        """
        INSERT INTO analysis_results
            (report_id, disaster_type, confidence_score, reliability_score,
             alert_level, location_lat, location_lon, analysis_data)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (report_id) DO UPDATE
           SET disaster_type     = EXCLUDED.disaster_type,
               confidence_score  = EXCLUDED.confidence_score,
               reliability_score = EXCLUDED.reliability_score,
               alert_level       = EXCLUDED.alert_level,
               location_lat      = EXCLUDED.location_lat,
               location_lon      = EXCLUDED.location_lon,
               analysis_data     = EXCLUDED.analysis_data,
               processed_at      = CURRENT_TIMESTAMP
        """,
        (
            report.report_id,
            report.disaster_type.value if report.disaster_type else "unknown",
            report.confidence_score,
            report.reliability_score,
            report.alert_level.value if report.alert_level else 1,
            report.location.get("lat", 0.0) if report.location else 0.0,
            report.location.get("lon", 0.0) if report.location else 0.0,
            Json(payload),
        ),
    )
    self.db_connection.commit()

# Add these methods to the class
OceanDisasterDetector.load_pending_reports = load_pending_reports # type: ignore
OceanDisasterDetector._load_images_for_report = _load_images_for_report # type: ignore
OceanDisasterDetector._update_report_status = _update_report_status # type: ignore
OceanDisasterDetector._save_analysis_result = _save_analysis_result #type: ignore

print("FIXED data loading methods with proper JSON serialization added to OceanDisasterDetector class!")

# Cell 5: Image Analysis Methods

def extract_gps_from_images(self, images: List[str]) -> Optional[Dict[str, float]]:
    if images is None:
        return None

    for image_data in images:
        try:
            if not image_data:
                continue

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            exifdata = image.getexif()

            if exifdata is not None:
                for tag_id in exifdata:
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_info = exifdata.get_ifd(tag_id)
                        gps_data = self._parse_gps_data(gps_info)
                        if gps_data:
                            return gps_data
        except Exception as e:
            print(f"Error extracting GPS from image: {e}")
            continue

    return None

def _parse_gps_data(self, gps_info) -> Optional[Dict[str, float]]:
    try:
        def convert_to_degrees(value):
            if not value or not isinstance(value, (list, tuple)) or safe_len(value) < 3:
                return None
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)

        lat = None
        lon = None

        if 2 in gps_info and 1 in gps_info:
            lat_degrees = convert_to_degrees(gps_info[2])
            if lat_degrees is not None:
                lat = lat_degrees
                if gps_info[1] == "S":
                    lat = -lat

        if 4 in gps_info and 3 in gps_info:
            lon_degrees = convert_to_degrees(gps_info[4])
            if lon_degrees is not None:
                lon = lon_degrees
                if gps_info[3] == "W":
                    lon = -lon

        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}

    except Exception as e:
        print(f"Error parsing GPS data: {e}")

    return None

def analyze_images(self, images: List[str]) -> Dict:
    analysis_results = {
        "total_images": 0,
        "disaster_indicators": [],
        "confidence": 0.0,
        "visual_features": [],
        "gps_extracted": False,
        "gps_location": None,
    }

    if images is None:
        return analysis_results

    analysis_results["total_images"] = safe_len(images)

    disaster_indicators = 0
    total_confidence = 0.0

    gps_location = self.extract_gps_from_images(images)
    if gps_location:
        analysis_results["gps_extracted"] = True
        analysis_results["gps_location"] = gps_location

    for i, image_data in enumerate(images):
        try:
            if not image_data:
                analysis_results["disaster_indicators"].append(
                    {"has_disaster": False, "confidence": 0.0, "error": "Empty image data"}
                )
                continue

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            features = self._extract_visual_features(image)
            analysis_results["visual_features"].append(features)

            indicators = self._detect_disaster_indicators(features)
            if indicators["has_disaster"]:
                disaster_indicators += 1
                total_confidence += indicators["confidence"]

            analysis_results["disaster_indicators"].append(indicators)

        except Exception as e:
            print(f"Error analyzing image {i}: {e}")
            analysis_results["disaster_indicators"].append(
                {"has_disaster": False, "confidence": 0.0, "error": str(e)}
            )

    if disaster_indicators > 0:
        analysis_results["confidence"] = total_confidence / disaster_indicators

    return analysis_results

def _extract_visual_features(self, image: Image) -> Dict: #type: ignore
    if not image:
        return {
            "mean_red": 0.0, "mean_green": 0.0, "mean_blue": 0.0,
            "texture": 0.0, "edge_strength": 0.0, "brightness": 0.0,
        }

    img_array = np.array(image.resize((224, 224))) #type: ignore

    if safe_len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    mean_rgb = np.mean(img_array, axis=(0, 1))
    gray = np.mean(img_array, axis=2)
    texture = np.std(gray)
    edges = np.sum(np.abs(np.gradient(gray)))

    return {
        "mean_red": float(mean_rgb[0]),
        "mean_green": float(mean_rgb[1]) if safe_len(mean_rgb) > 1 else float(mean_rgb[0]),
        "mean_blue": float(mean_rgb[2]) if safe_len(mean_rgb) > 2 else float(mean_rgb[0]),
        "texture": float(texture),
        "edge_strength": float(edges),
        "brightness": float(np.mean(gray)),
    }

def _detect_disaster_indicators(self, features: Dict) -> Dict:
    if not features or not isinstance(features, dict):
        return {
            "has_disaster": False,
            "confidence": 0.0,
            "predicted_type": DisasterType.UNKNOWN,
        }

    confidence = 0.0
    disaster_type = DisasterType.UNKNOWN

    brightness = features.get("brightness", 0)
    texture = features.get("texture", 0)
    edge_strength = features.get("edge_strength", 0)

    if brightness < 100 and texture > 50:
        confidence += 0.3
        disaster_type = DisasterType.TSUNAMI

    if edge_strength > 10000:
        confidence += 0.2

    if (features.get("mean_red", 0) > features.get("mean_blue", 0) and
        features.get("mean_green", 0) > features.get("mean_blue", 0)):
        confidence += 0.2
        disaster_type = DisasterType.COASTAL_FLOODING

    if brightness < 80 and texture < 30:
        confidence += 0.4
        disaster_type = DisasterType.OIL_SPILL

    return {
        "has_disaster": confidence > 0.3,
        "confidence": min(confidence, 1.0),
        "predicted_type": disaster_type,
    }

# Add methods to class
OceanDisasterDetector.extract_gps_from_images = extract_gps_from_images #type: ignore
OceanDisasterDetector._parse_gps_data = _parse_gps_data #type: ignore
OceanDisasterDetector.analyze_images = analyze_images #type: ignore
OceanDisasterDetector._extract_visual_features = _extract_visual_features #type: ignore
OceanDisasterDetector._detect_disaster_indicators = _detect_disaster_indicators #type: ignore

print("Image analysis methods added to OceanDisasterDetector class!")

# Cell 6: Text Analysis and Risk Assessment Methods

def analyze_description(self, description: str) -> Tuple[DisasterType, float]:
    if description is None or not isinstance(description, str):
        return DisasterType.UNKNOWN, 0.0

    desc_lower = description.lower().strip()
    if not desc_lower:
        return DisasterType.UNKNOWN, 0.0

    best_type = DisasterType.UNKNOWN
    best_score = 0.0

    for dtype, keywords in self.disaster_keywords.items():
        if keywords is None or not isinstance(keywords, list):
            continue

        matches = sum(1 for kw in keywords if kw and isinstance(kw, str) and kw in desc_lower)
        score = matches / max(safe_len(keywords), 1) if keywords else 0.0

        if score > best_score:
            best_type = dtype
            best_score = score

    return best_type, best_score

def calculate_reliability_score(
    self,
    description: str,
    selected_disaster: DisasterType,
    image_analysis: Dict,
    location: Optional[Dict[str, float]],
) -> float:
    score = 0.0

    if description and isinstance(description, str) and description.strip():
        desc_score = min(safe_len(description.split()) / 20.0, 1.0)
        score += desc_score * 0.3

    if image_analysis and isinstance(image_analysis, dict) and image_analysis.get("confidence", 0) > 0:
        score += image_analysis["confidence"] * 0.4

    if location and isinstance(location, dict) and "lat" in location and "lon" in location:
        if self._is_coastal_location(location):
            score += 0.2
        else:
            score += 0.1

    if location and isinstance(location, dict):
        historical_score = self._check_historical_consistency(selected_disaster, location)
        score += historical_score * 0.1

    return min(score, 1.0)

def _is_coastal_location(self, location: Dict[str, float]) -> bool:
    if not location or "lat" not in location or "lon" not in location:
        return True
    cur = self.db_connection.cursor()
    try:
        cur.execute(
            """
            SELECT is_coastal
              FROM locations
             WHERE ABS(latitude - %s) < 0.5
               AND ABS(longitude - %s) < 0.5
               AND is_coastal IS NOT NULL
             ORDER BY id
             LIMIT 1
            """,
            (location["lat"], location["lon"]),
        )
        row = cur.fetchone()
        if row is not None:
            return bool(row[0])
    except Exception:
        pass
    return True

def _check_historical_consistency(self, disaster_type: DisasterType, location: Dict[str, float]) -> float:
    if not location or not disaster_type:
        return 0.3
    cur = self.db_connection.cursor()
    try:
        cur.execute(
            """
            SELECT disaster_type, COUNT(*) AS cnt
              FROM historical_disasters
             WHERE ABS(latitude - %s) < 1.0
               AND ABS(longitude - %s) < 1.0
             GROUP BY disaster_type
            """,
            (location.get("lat", 0.0), location.get("lon", 0.0)),
        )
        rows = cur.fetchall()
        total = sum(c for _, c in rows)
        if total == 0:
            return 0.5
        for db_type, c in rows:
            if db_type == disaster_type.value:
                return c / total
    except Exception:
        pass
    return 0.3

def determine_alert_level(
    self,
    disaster_type: DisasterType,
    confidence: float,
    reliability: float,
    image_analysis: Dict,
) -> AlertLevel:
    if not disaster_type:
        disaster_type = DisasterType.UNKNOWN

    confidence = confidence if isinstance(confidence, (int, float)) else 0.0
    reliability = reliability if isinstance(reliability, (int, float)) else 0.0
    image_analysis = image_analysis if isinstance(image_analysis, dict) else {}

    severity_mapping = {
        DisasterType.TSUNAMI: 3,
        DisasterType.HURRICANE: 2,
        DisasterType.STORM_SURGE: 2,
        DisasterType.COASTAL_FLOODING: 1,
        DisasterType.RIP_CURRENT: 1,
        DisasterType.HIGH_WAVES: 1,
        DisasterType.ALGAE_BLOOM: 1,
        DisasterType.OIL_SPILL: 2,
        DisasterType.EROSION: 1,
        DisasterType.UNKNOWN: 1,
    }

    base_severity = severity_mapping.get(disaster_type, 1)
    combined_score = (confidence + reliability) / 2

    if combined_score < 0.4:
        adjusted_severity = max(1, base_severity - 1)
    elif combined_score > 0.8:
        if base_severity < 3 and disaster_type in [DisasterType.HURRICANE, DisasterType.STORM_SURGE]:
            adjusted_severity = base_severity + 1
        else:
            adjusted_severity = base_severity
    else:
        adjusted_severity = base_severity

    if image_analysis and image_analysis.get("confidence", 0) > 0.7:
        adjusted_severity = min(3, adjusted_severity + 1)

    return AlertLevel(adjusted_severity)

def _get_recommended_actions(
    self, disaster_type: DisasterType, alert_level: AlertLevel
) -> List[str]:
    if not disaster_type:
        disaster_type = DisasterType.UNKNOWN
    if not alert_level:
        alert_level = AlertLevel.WARNING

    base_actions = {
        DisasterType.TSUNAMI: {
            AlertLevel.WARNING: ["Monitor official alerts", "Review evacuation routes"],
            AlertLevel.EMERGENCY: ["Move to higher ground immediately", "Stay away from beaches"],
            AlertLevel.EVACUATION: ["EVACUATE IMMEDIATELY to high ground", "Do not return until all-clear"],
        },
        DisasterType.HURRICANE: {
            AlertLevel.WARNING: ["Secure outdoor items", "Stock emergency supplies"],
            AlertLevel.EMERGENCY: ["Stay indoors", "Avoid coastal areas"],
            AlertLevel.EVACUATION: ["EVACUATE coastal and flood-prone areas", "Follow official evacuation orders"],
        },
        DisasterType.STORM_SURGE: {
            AlertLevel.WARNING: ["Monitor weather updates", "Prepare for possible flooding"],
            AlertLevel.EMERGENCY: ["Move away from low-lying areas", "Secure property"],
            AlertLevel.EVACUATION: ["EVACUATE flood-prone areas immediately", "Seek higher ground"],
        },
        DisasterType.OIL_SPILL: {
            AlertLevel.WARNING: ["Avoid affected water areas", "Report wildlife impacts"],
            AlertLevel.EMERGENCY: ["Do not enter contaminated areas", "Seek medical attention if exposed"],
            AlertLevel.EVACUATION: ["EVACUATE immediately if fumes present", "Follow health authority guidance"],
        },
    }

    default_actions = {
        AlertLevel.WARNING: ["Stay alert", "Monitor official channels"],
        AlertLevel.EMERGENCY: ["Take immediate precautions", "Avoid affected areas"],
        AlertLevel.EVACUATION: ["EVACUATE if instructed by authorities"],
    }

    return base_actions.get(disaster_type, default_actions).get(
        alert_level, default_actions[AlertLevel.WARNING]
    )

# Add methods to class
OceanDisasterDetector.analyze_description = analyze_description #type: ignore
OceanDisasterDetector.calculate_reliability_score = calculate_reliability_score #type: ignore
OceanDisasterDetector._is_coastal_location = _is_coastal_location #type: ignore
OceanDisasterDetector._check_historical_consistency = _check_historical_consistency #type: ignore
OceanDisasterDetector.determine_alert_level = determine_alert_level #type: ignore
OceanDisasterDetector._get_recommended_actions = _get_recommended_actions #type: ignore

print("Text analysis and risk assessment methods added to OceanDisasterDetector class!")

# Cell 7: Report Processing and Schema Creation

def create_schema(self, report: DisasterReport) -> Dict:
    if not report:
        return {}

    return {
        "event_id": report.report_id or f"{report.disaster_type.value}_{int(report.timestamp.timestamp())}",
        "disaster_type": report.disaster_type.value,
        "alert_level": report.alert_level.value,
        "alert_name": report.alert_level.name,
        "confidence_score": round(report.confidence_score, 3),
        "reliability_score": round(report.reliability_score, 3),
        "location": report.location or {"lat": 0.0, "lon": 0.0},
        "timestamp": report.timestamp.isoformat(),
        "description": report.description or "",
        "image_analysis": report.image_analysis or {},
        "risk_assessment": {
            "immediate_danger": report.alert_level.value >= 2,
            "evacuation_needed": report.alert_level.value == 3,
            "monitoring_required": True,
        },
        "recommended_actions": self._get_recommended_actions(report.disaster_type, report.alert_level),
    }

def process_report(self, data: Dict) -> DisasterReport:
    if not data or not isinstance(data, dict):
        data = {}

    description = data.get("description", "") or ""
    images = data.get("images", []) or []
    selected_disaster = data.get("selected_disaster")
    provided_location = data.get("location")
    report_id = data.get("report_id")

    # Ensure images is never None
    if images is None:
        images = []

    location = None
    image_analysis: Dict[str, Any] = {}

    if images and isinstance(images, list):
        image_analysis = self.analyze_images(images)
        if image_analysis.get("gps_extracted") and image_analysis.get("gps_location"):
            location = image_analysis["gps_location"]

    if not location and provided_location and isinstance(provided_location, dict):
        location = provided_location

    if not location:
        location = {"lat": 0.0, "lon": 0.0}

    if selected_disaster:
        try:
            disaster_type = DisasterType(selected_disaster)
            desc_confidence = 1.0 if description and description.strip() else 0.5
        except ValueError:
            disaster_type = DisasterType.UNKNOWN
            desc_confidence = 0.3
    else:
        disaster_type, desc_confidence = self.analyze_description(description)

    confidence_score = desc_confidence
    if image_analysis and isinstance(image_analysis, dict) and "confidence" in image_analysis:
        confidence_score = (confidence_score + image_analysis["confidence"]) / 2

    reliability_score = self.calculate_reliability_score(
        description, disaster_type, image_analysis, location
    )

    alert_level = self.determine_alert_level(
        disaster_type, confidence_score, reliability_score, image_analysis
    )

    report = DisasterReport(
        disaster_type=disaster_type,
        confidence_score=confidence_score,
        reliability_score=reliability_score,
        alert_level=alert_level,
        location=location,
        timestamp=datetime.now(),
        description=description,
        image_analysis=image_analysis,
        schema={},
        report_id=report_id,
    )

    report.schema = self.create_schema(report)
    return report

def process_all_reports(self) -> List[DisasterReport]:
    print("Loading pending reports from database...")
    pending = self.load_pending_reports()
    if not pending:
        print("No pending reports found in database")
        return []
    print(f"Found {len(pending)} pending reports")
    processed: List[DisasterReport] = []
    for i, report_data in enumerate(pending):
        try:
            rid = report_data.get("report_id", "unknown")
            print(f"Processing report {i+1}/{len(pending)}: {rid}")
            report = self.process_report(report_data)
            processed.append(report)
            self._update_report_status(rid, "processed")
            self._save_analysis_result(report)
        except Exception as e:
            print(f"Error processing report {report_data.get('report_id', 'unknown')}: {e}")
            self._update_report_status(report_data.get("report_id"), "error", str(e))
    return processed

# Add methods to class
OceanDisasterDetector.create_schema = create_schema #type: ignore
OceanDisasterDetector.process_report = process_report #type: ignore
OceanDisasterDetector.process_all_reports = process_all_reports #type: ignore

print("Report processing methods added!")

# Cell 8 FIXED: Output and Main Analysis Functions with proper JSON serialization

def save_results_to_json(self, reports: List[DisasterReport]) -> str:
    """FIXED: Save results with proper enum serialization"""
    if reports is None:
        reports = []

    output_data: Dict[str, Any] = {
        "total_reports": safe_len(reports),
        "summary": {
            "by_disaster_type": {},
            "by_alert_level": {},
            "average_confidence": 0.0,
            "average_reliability": 0.0,
        },
        "reports": [],
    }

    total_confidence = 0.0
    total_reliability = 0.0

    for report in reports:
        if not report:
            continue

        try:
            # Use the fixed to_dict method that handles enum serialization
            report_dict = report.to_dict()
            output_data["reports"].append(report_dict)

            # Get serialized values instead of enum objects
            disaster_type = report.disaster_type.value if report.disaster_type else "unknown"
            alert_level = report.alert_level.name if report.alert_level else "UNKNOWN"

            if disaster_type not in output_data["summary"]["by_disaster_type"]:
                output_data["summary"]["by_disaster_type"][disaster_type] = 0
            if alert_level not in output_data["summary"]["by_alert_level"]:
                output_data["summary"]["by_alert_level"][alert_level] = 0

            output_data["summary"]["by_disaster_type"][disaster_type] += 1
            output_data["summary"]["by_alert_level"][alert_level] += 1

            total_confidence += getattr(report, "confidence_score", 0.0)
            total_reliability += getattr(report, "reliability_score", 0.0)
        except Exception as e:
            print(f"Error processing report for JSON output: {e}")
            continue

    if reports:
        output_data["summary"]["average_confidence"] = round(total_confidence / safe_len(reports), 3)
        output_data["summary"]["average_reliability"] = round(total_reliability / safe_len(reports), 3)

    try:
        # FIXED: Use custom JSON encoder to handle any remaining enum issues
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, cls=DisasterJSONEncoder)
        print(f"Results saved to {self.output_path}")
        return self.output_path
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
        backup_path = "backup_disaster_results.json"
        try:
            # FIXED: Use custom JSON encoder for backup too
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, cls=DisasterJSONEncoder)
            print(f"Results saved to backup location: {backup_path}")
            return backup_path
        except Exception as e2:
            print(f"Failed to save to backup location: {e2}")
            return ""

def run_complete_analysis(self) -> Optional[str]:
    print("Starting Ocean Disaster Detection Analysis...")
    print(f"Output: {self.output_path}")
    try:
        reports = self.process_all_reports()
        if not reports:
            print("No reports were processed.")
            return None
        output_file = self.save_results_to_json(reports)
        if output_file:
            print("Analysis complete.")
            return output_file
        print("Analysis completed but failed to save results.")
        return None
    except Exception as e:
        print(f"Fatal error during complete analysis: {e}")
        return None

# Add methods to class
OceanDisasterDetector.save_results_to_json = save_results_to_json #type: ignore
OceanDisasterDetector.run_complete_analysis = run_complete_analysis #type: ignore

detector = OceanDisasterDetector()
detector.run_complete_analysis() # type: ignore
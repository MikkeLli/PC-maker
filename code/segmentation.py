import cv2
import numpy as np
from dataclasses import dataclass
from patient import Patient

"""SegmentationController -- оркестратор
│
├── SegmentationSession -- dataclass со всеми ROI&data
│
├── Settings -- dataclass настроек
│
├── Detector -- base class ← поиск начальной ROI
│     ├── MagnitudeDetector
│     ├── VelocityDetector
│     ├── TemporalDetector
│     └── HybridDetector
│
├── Tracker -- base class ← перенос ROI между кадрами
│     ├── PropagationTracker
│     ├── OpticalFlowTracker
│     └── RigidTracker
│
├── Refinement  ← морфологическая обработка и уточнение
│     ├── MorphologyProcessor
│     ├── ContourSmoother
│     └── HoleFiller
│
├── Validation ← контроль качества
│     ├── GeometryValidator
│     ├── TemporalValidator
│     └── QualityEstimator
│
├── Editing ← ручное редактирование
│     ├── BrushTool
│     ├── PolygonTool
│     ├── SmartCorrection
│     └── UndoManager
│
└── Metrics ← вычисление геометрических и гемодинамических параметров
      ├── GeometryMetrics
      ├── FlowMetrics
      └── ExportMetrics"""
@dataclass
class Settings:
      desired_roi: str = None
@dataclass
class ROIMetrics:
      area: float = None
      perimeter: float = None
      circularity: float = None
      mean_velocity: float = None
      peak_velocity: float = None

@dataclass
class SegmentationSession:
      patient: Patient = None # or dataset
      roi = None
      settings: Settings = None
      metrics: Metrics = None
      desired_roi: str = None

class SegmentationController:
      def __init__(self, patient: Patient, settings: Settings):
            self.patient = patient
            self.settings = settings
            self.session = SegmentationSession(patient=patient) # TODO: вынести в метод вычислительных классов

      def run_segmentation(self):
        # Step 1: Detect initial ROI
            detector = MagnitudeDetector(self.settings)
            initial_roi = detector.detect(self.session)
            self.session.roi = initial_roi

        # Step 2: Refine the ROIs
            refiner = MorphologyProcessor(tracked_rois)
            refined_rois = refiner.refine()
            self.session.rois = refined_rois

        # Step 3: Temporal analysis of ROis
            temporal_analyzer = TemporalAnalyzer(refined_rois)
            temporal_results = temporal_analyzer.analyze()

        # Step 4: ROI tracking across frames if needed 

        # Step 5: classify ROIs
            classifier = ROIClassifier(refined_rois)
            classified_rois = classifier.classify()

        # Step 6: Validate the ROis manually or automatically
            validator = GeometryValidator(classified_rois)
            
            return roi_mask, roi_metrics
    
class Detector:
      def __init__(self, settings: Settings):
            self.settings = settings
            
      def detect(self):
            raise NotImplementedError("Detector subclasses must implement the detect method.")

class MagnitudeDetector(Detector):
      def __init__(self, settings: Settings):
            super().__init__(settings)

      def detect(self, session: SegmentationSession):
            # Implement detection logic based on magnitude images
            # For example, thresholding or edge detection
            patient = session.patient
            magnitude_image = patient.magnitude_image

            if session.desired_roi == "aqueduct":
                  pass

            return roi


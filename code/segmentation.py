from typing import Generator, Iterator, List

import cv2
import numpy as np
from dataclasses import dataclass
from patient import Patient
import config
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
      metrics: ROIMetrics = None
      desired_roi: str = None
      candidate_rois = None

class SegmentationController:
      def __init__(self, patient: Patient, settings: Settings):
            self.patient = patient
            self.settings = settings
            self.session = SegmentationSession(patient=patient) # TODO: вынести в метод вычислительных классов

      def run_segmentation(self):
        # Step 1: Detect initial ROI
            detector = MagnitudeDetector(self.settings)
            candidate_rois = detector.detect(self.session)
            self.candidate_rois = candidate_rois



      #   # Step 4: ROI tracking across frames if needed 

      #   # Step 5: classify ROIs
      #       classifier = ROIClassifier(refined_rois)
      #       classified_rois = classifier.classify()

      #   # Step 6: Validate the ROis manually or automatically
      #       validator = GeometryValidator(classified_rois)
            
            return None
    
class Detector:
      def __init__(self, settings: Settings):
            self.settings = settings
            self.candidate_rois = None

      def detect(self):
            raise NotImplementedError("Detector subclasses must implement the detect method.")

      def morphology_refinement(self):
            raise NotImplementedError("Detector subclasses must implement the morphology_refinement method.")
class MagnitudeDetector(Detector):
      def __init__(self, settings: Settings):
            super().__init__(settings)

      def detect(self, session: SegmentationSession):
            # Implement detection logic based on magnitude images
            # For example, thresholding or edge detection
            patient = session.patient
            magnitude_image = patient.magnitude_image[:,:,0] #.astype(np.uint8)
            
            blurred = cv2.GaussianBlur(magnitude_image, (7, 7), 0)
            raw_binary_mask = cv2.threshold(blurred, 220,255,cv2.THRESH_BINARY)[1].astype(np.uint8)
            
            # cv2.imshow("Magnitude Detector Output", raw_binary_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            analyze = cv2.connectedComponentsWithStats(raw_binary_mask, 4, cv2.CV_32S)
            (num_labels, labels, stats, centroids) = analyze

            candidate_rois = np.zeros(raw_binary_mask.shape, dtype=np.uint8)
            for i in range(1, num_labels):
                  area = stats[i, cv2.CC_STAT_AREA]
                  centroid = centroids[i]
                  def _is_center(centroid, image_shape):
                        center_y = image_shape[0] / 2
                        pad = 0.05 * image_shape[0]  # 10% padding
                        return (center_y - pad) <= centroid[1] <= (center_y + pad)
                  if area > 50 and _is_center(centroid, raw_binary_mask.shape):
                        component_mask = (labels == i).astype(np.uint8) * 255
                        candidate_rois = cv2.bitwise_or(candidate_rois, component_mask)
                   
            cv2.imshow("Candidate ROIs", candidate_rois)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.candidate_rois = candidate_rois
            return candidate_rois    

      def morphology_refinement(self):
            pass #не понятно, стоит ли делать такое

class FeatureAnalyzer:
      def __init__(self, settings: Settings):
            self.settings = settings

      def extract_features(self, session: SegmentationSession):
            # return сomponent - Geom, Temp, QC
            pass

      def temporal_analyze(self, session: SegmentationSession):
            patient = session.patient
            
            candidates_mask = session.candidate_rois
            velocity_map = patient.velocity_map
            moving_metrics = self._compute_moving_metrics(velocity_map, candidates_mask)
            #TODO: добавить расчет этой метрики для каждой компоненты, может сделать поле или метод, который 
            # будет давать результат connectedComponentsWithStats
            return moving_metrics


      def geometry_analyze(self, session: SegmentationSession):
            patient = session.patient
            candidates_mask = session.candidate_rois

            pass

      def _compute_moving_metrics(self, velocity_map, mask) -> Iterator[tuple[tuple[float, float], dict]]:
            """
            Рассчитывает карту показателя периодического движения - амплитуду
            основной гармоники для каждого пикселя, принадлежащему области интереса"""
            for i in range(velocity_map.shape[1]):
                  for j in range(velocity_map.shape[2]):
                        if mask[i, j]:
                              pixel_time_series = velocity_map[:, i, j]
                              # Perform FFT
                              fft_result = np.fft.fft(pixel_time_series)
                              # Compute amplitude of the fundamental frequency
                              fundamental_amplitude = np.abs(fft_result[1])  # Assuming the first index is DC component
                              yield (i, j), {'FFT1':fundamental_amplitude, 'mean_velocity': np.mean(pixel_time_series),
                                              'peak_velocity': np.abs(np.max(pixel_time_series))}



Detector = MagnitudeDetector(Settings())
patient_path = config.PATIENTS_DIR / "Aqueduct"
patient = Patient(patient_path)
patient.load_dicom()
SegmentationSession = SegmentationSession(patient=patient, settings=Settings())
Detector.detect(SegmentationSession)
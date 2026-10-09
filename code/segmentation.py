import cv2
import numpy as np

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
from preprocessing import BackGroundPhaseCorrector, AliasingCorrector, PreprocessConfig
from patient import Patient

def correct_background(patient, use_robust: bool = False, iterations: int = 1,
                       qc: bool = True, qc_view: bool = False):
    raw_phase = patient.raw_phase
    initial_mask = patient.background_mask

    Corrector = BackGroundPhaseCorrector(raw_phase, initial_mask)
    Corrector.fit()

    if use_robust:
        Corrector.correct(iterations=iterations)

    if qc:
        Corrector.compute_qc(view=qc_view)

    corrected_phase = Corrector.predict()

    return corrected_phase



patient_path = r"/Patients/Aqueduct"
config = PreprocessConfig(temporal_std_threshold_q=600.0, magnitude_threshold_q=300)
# возвращаем (y, x, t)
test_patient = Patient(patient_path)

test_patient.load_dicom()
test_patient.make_background_mask(config.temporal_std_threshold_q, config.magnitude_threshold_q)


test_patient.corrected_phase = correct_background(test_patient, qc=True) # the shape is inverted to raw_phase

Corrector = AliasingCorrector(test_patient)

test_patient.corrected_phase, _ = Corrector.temporal_unwrap(view=True)



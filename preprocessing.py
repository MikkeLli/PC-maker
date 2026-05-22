import re
from pathlib import Path
import json
import numpy
import nibabel as nib
import dicom2nifti
import dcm2niixpy
import subprocess
from dataclasses import dataclass
import os
from dcm2niix import main
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Converts all DICOM series in the folder to separate NIfTI files
# dcm2niix = dcm2niixpy.DCM2NIIX(version="1.0.20220720")

"""Этот файл осуществляет предобработку данных PC-MRI:
-- Сортировка и QA входных серий.
    Распозование пары/наборы phase–magnitude, cardiac phases, значение VENC, направление кодирования. 
-- Коррекция background phase / velocity offset.
    image-based коррекции на основе локального фона, полиномиальной аппроксимации по фону, а также методы с phantom/field monitoring;
-- Коррекция aliasing / phase unwrapping.
    Поэтому либо подбирают VENC заранее, либо после реконструкции применяют phase unwrapping, если aliasing уже случился. 
    Для бейзлайна я бы сделал авто-детект aliasing и ручной режим разворачивания как минимум.
-- Шумоподавление и подавление выбросов.
    Поэтому обычно используют мягкую фильтрацию, маскирование по magnitude и ручной контроль выбросов
"""
@dataclass
class PreprocessConfig:
    background_poly_order: int = 2
    use_robust_fit: bool = True
    sigma_clip: float = 3.0
    magnitude_threshold_q: float = 0.2
    temporal_std_threshold_q: float = 0.3
    denoise_preview_sigma: float = 0.8
    unwrap_aliasing: bool = True


patient_path = r"D:\Program_Files\ML\PC-MRI\Patients\Aqueduct"


def load_dicom(dicom_dir: str | os.PathLike[str]) -> tuple:

    """Загрузка dicom папки, возврат phase, magnitude, metadata dict из настроек"""
    if not any(Path(dicom_dir).glob("*ph*")):
        # dcm2niix -i y -z y ./
        # dcm2niix.ignore_derived = 'y'
        # dcm2niix.verbose = 2
        # dcm2niix.conflict_write_behavior = 0
        # dcm2niix.convert(dicom_dir, dicom_dir)
        # Basic command line call via Python
        # -z y: compress to .nii.gz
        # -o: output directory
        # 'path/to/dicom': input folder
        # Run with 60 second timeout
        try:
            exit_code = main(
                ["-z", "y", "-v", "1", '-o', dicom_dir, dicom_dir]
            )

            if exit_code == 0:
                print("Success: All files converted")
                status = "success"
            elif exit_code == 2:
                print("Error: No valid DICOM files found")
                status = "no_dicoms"
            elif exit_code == 4:
                print("Error: Corrupt DICOM file detected")
                status = "corrupt"
            elif exit_code == 5:
                print("Error: Input folder invalid")
                status = "invalid_input"
            elif exit_code == 6:
                print("Error: Output folder invalid")
                status = "invalid_output"
            elif exit_code == 7:
                print("Error: Output folder is read-only")
                status = "readonly"
            elif exit_code == 8:
                print("Warning: Partial success - some files converted")
                status = "partial"
            elif exit_code == 10:
                print("Warning: Incomplete volumes detected")
                status = "incomplete"
            else:
                print(f"Error: Conversion failed with exit code {exit_code}")
                status = "failed"
        except Exception as e:
            print(e)

    phase_img = None
    magn_img = None
    VENC = None
    for file in Path(dicom_dir).glob('*ph.nii*'):
        phase_img = nib.load(os.path.join(dicom_dir, file))
    for file in Path(dicom_dir).glob('*.json'):
        with open(file, 'r') as f:
            metadata = json.load(f)
        if "MAG" in metadata["SeriesDescription"]:
            magn_img = nib.load(os.path.join(dicom_dir, file.with_suffix('.nii.gz')))
        VENC = int(re.search(r'(\d+)', metadata.get("ProtocolName"))[0])
        # print(VENC)
    if not phase_img or not magn_img:
        raise ValueError("There's no phase or magnitude image")
    phase_img_data = phase_img.get_fdata()
    magn_img_data = magn_img.get_fdata()
    shape = phase_img_data.shape
    img_data = {}
    if phase_img_data.shape != magn_img_data.shape:
        raise ValueError("The Phase and magnitude images have different sizes")
    img_data['NativeShape'] = phase_img_data.shape
    if len(shape) == 2:
        img_data["Type"] = "2D"
    elif len(shape) == 3:
        if shape[2] == 1:
            img_data["Type"] = "2D"
        else:
            img_data["Type"] = "3D"
    elif len(shape) == 4:
        if shape[3] <= 3:
            img_data["Type"] = "3D"
            phase_img_data = phase_img_data[:,:,:, 1]
            magn_img_data = magn_img_data[:,:,:, 1]
        elif shape[2] == 1:
            img_data["Type"] = "2D+t"
            phase_img_data = phase_img_data[:, :, 0, :]
            magn_img_data = magn_img_data[:, :, 0, :]
        else:
            img_data["Type"] = "4D"
    else:
        raise TypeError("Unknown image shape")
    img_data['Shape'] = phase_img_data.shape
    img_data['VENC'] = VENC
    # print(img_data)
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 10))
    axes[0].imshow(magn_img_data[:, :, 0].T, cmap="gray", origin="lower")
    axes[1].imshow(phase_img_data[:, :, 0].T, cmap="gray", origin="lower")
    plt.show()
    return phase_img_data, magn_img_data, img_data

def background_phase_correction(phase: np.ndarray, magn: np.ndarray, dim: str, std_thres, magn_thres):
    # TODO: сделать непосредственно коррекцию
    if dim == "4D" or dim == "2D+t":
        # TODO: temproral_std
        magn_slice = magn[:,:,0]
        std = np.std(phase, axis=2, ddof=1)
        # print(std[310,310] < std_thres)
        background_mask_std = (std < std_thres)
        background_mask_magn = magn_slice < magn_thres
        background_mask = (background_mask_std * background_mask_magn)
        plt.imshow(background_mask_std.T, cmap="gray", origin="lower")
        plt.show()
    elif dim == "3D" or dim == "2D":
        threshold = np.percentile(magn, 20)
        background_mask = magn > threshold
    else:
        raise ValueError("Dim is not correct")
    return
config = PreprocessConfig(temporal_std_threshold_q=1100.0, magnitude_threshold_q=1000)
phase, magn, metadata = load_dicom(patient_path)
background_phase_correction(phase, magn, metadata['Type'], config.temporal_std_threshold_q, config.magnitude_threshold_q)
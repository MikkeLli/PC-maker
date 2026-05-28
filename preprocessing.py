import re
from pathlib import Path
import json
from pickletools import uint8

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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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
    background_fit_model = "polynom"
    sigma_clip: float = 3.0
    magnitude_threshold_q: float = 500
    temporal_std_threshold_q: float = 400
    denoise_preview_sigma: float = 0.8
    unwrap_aliasing: bool = True

class BackGroundPhaseCorrector:

    def __init__(self, phase: np.ndarray, mask: np.ndarray, poly_order: int =2, model: str ='polynom', robust: bool =True, qc: bool=True):
        self.background_coords = None
        self.poly_order = poly_order
        self.model_type = model
        self.use_robust = robust
        self.use_qc = qc
        self.phase = phase
        self.mask = mask
        self.model = None
        self.shape = self.phase.shape
        self.fit_mask = None

    def fit(self):
        # TODO: добавить параметры фита в аргументы
        # coef = ['a','b','c','d','e','f','g','h','k','m','n']
        if self.model_type == 'polynom':
            if self.poly_order == 2:
                def curve(x, a, b, c, d, e, f):
                    return a + b*x[0] + c*x[1] + d*x[0]*x[1] + e * x[0] * x[0] + f * x[1] * x[1]
            else:
                raise ValueError("This polyorder is currently unsupported")
        else:
            raise ValueError("Incorrect model type")
        # shape = self.phase.shape
        # limits = [0, shape[0], 0, shape[1]]  # [x1_min, x1_max, x2_min, x2_max]
        # side_x = np.linspace(limits[0], limits[1], shape[0])
        # side_y = np.linspace(limits[2], limits[3], shape[1])
        # X, Y = np.meshgrid(side_x, side_y)
        # size = X.shape
        # x_1d = X.reshape((1, np.prod(size)))
        # y_1d = Y.reshape((1, np.prod(size)))
        #
        # xdata = np.vstack((x_1d, y_1d))
        if self.fit_mask is None:
            mask = self.mask
        else:
            mask = self.fit_mask
        self.background_coords = np.argwhere(mask == 1)

        param_in_time = np.ndarray(shape=(self.shape[2], 6))
        print("Performing fitting...")
        for t in range (self.shape[2]):  # цикл по фреймам
            time_slice = self.phase[:,:,t]
            ydata = time_slice[mask]
            ydata.astype(np.float64)
            coords_for_fit = self.background_coords.T.astype(np.float64)
            popt, pcov = curve_fit(curve, coords_for_fit, ydata, maxfev=50000, method='trf')

            param_in_time[t] = popt
        print("Fit done\n")
        # сглаживание параметров во времени
        print("Smoothing parameters...")
        param_in_time = param_in_time.T
        smoothed_params = np.ndarray(shape=param_in_time.shape)
        for i, param in enumerate(param_in_time):
            smoothed_param = savgol_filter(param, window_length=5, polyorder=3)
            smoothed_params[i] = smoothed_param
        self.model = (curve, smoothed_params.T)
        print("Done\n")
        return self.model

    def predict(self):
        #unpack model
        fit_func, params = self.model
        print("Predicting...")

        corrected_phase = np.ndarray(shape=(self.shape[2], self.shape[0], self.shape[1]))
        # предикт фазы для всего объема
        for t in range(self.shape[2]):
            time_slice = self.phase[:,:,t]

            predicted = np.asarray([[fit_func((i,j), *params[t]) for j in range(self.shape[1])] for i in range(self.shape[0])])
            corrected = time_slice - predicted
            corrected_phase[t] = corrected
        print("Done\n")
        return corrected_phase

    def correct(self, iterations: int =3):
        print("Correcting the fit...")
        if self.fit_mask is None:
            self.fit_mask = self.mask.copy()
        bad_count = 0
        for i in range(iterations):
            fit_func, params0 = self.model

            # TODO: написать клип по сигма для пикселей фона, исключить выбросы из маски фона, обновить self фон (копией), запустить новый фит с новым фоном
            background_slice = self.phase[:,:,0][self.fit_mask]
            # background_slice[~self.mask] = np.nan # ?
            predicted = np.asarray([fit_func(xy, *params0[0]) for xy in self.background_coords])
            residual = background_slice - predicted

            # need to be upgraded
            sigma = 1.4826 * np.median(abs(residual - np.median(residual)))

            # res_mask = abs(residual) > 3*sigma
            bad_idx = np.argwhere(abs(residual) > 3*sigma)
            bad_count_n = len(bad_idx)
            if np.isclose(bad_count_n, bad_count, atol=10):
                break
            else:
                bad_count = bad_count_n

            for idx in bad_idx:

                self.fit_mask[*self.background_coords[*idx]] = 0
                    # self.background_coords.pop(idx)
            self.fit()
        print("Done\n")
    def compute_qc(self, view: bool = False):

        old_slice = self.phase[:,:,0][self.mask]
        new_slice = self.phase[:,:, 0][self.fit_mask]
        old_background_mean = np.mean(old_slice)
        new_background_mean = np.mean(new_slice)

        old_background_std = np.std(old_slice, ddof=1)
        new_background_std = np.std(new_slice, ddof=1)

        print(f"Mean in the background before/after correction: {old_background_mean:.3f} / {new_background_mean:.3f}")
        print(f"Std in the background before/after correction: {old_background_std:.3f} / {new_background_std:.3f}")
        if view:
            fig, ax = plt.subplots(2,1, figsize=(12,12), sharex=True)
            ax[0].imshow(self.mask)
            ax[1].imshow(self.fit_mask)
            plt.show()
            plt.close(fig)

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
    magn_slice = magn_img_data[:, :, 0]
    phase_slice = phase_img_data[:, :, 0]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(magn_slice.T, cmap="gray", origin="lower")
    axes[0, 1].imshow(phase_slice.T, cmap="gray", origin="lower")
    # std = np.std(phase_img_data, axis=2, ddof=1)
    magn_for_hist = magn_slice.ravel()
    # phase_for_hist = phase_slice.ravel()
    axes[1,0].hist(magn_for_hist, density=True, bins=24)
    # axes[1,1].hist(phase_for_hist, density=False, bins=64)
    # plt.show()
    plt.savefig("test.jpg")
    plt.close()
    return phase_img_data, magn_img_data, img_data

def make_background_mask(phase: np.ndarray, magn: np.ndarray, dim: str, std_thres, magn_thres):
    if dim == "4D" or dim == "2D+t":
        # Берем диспресию фазы и первый фрейм магнитуды
        magn_slice = magn[:,:,0]
        std = np.std(phase, axis=2, ddof=1)
        # print(std[310,310] < std_thres)
        background_mask_std = (std < std_thres)
        background_mask_magn = magn_slice < magn_thres
        background_mask = (background_mask_std * background_mask_magn)
        kernel = np.ones((7, 7), np.uint8)

        # Perform morphological closing
        dilated_mask = cv2.erode(background_mask.astype(np.uint8), kernel)
        closed_mask = cv2.morphologyEx(background_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].imshow(background_mask.T, cmap="gray", origin="lower")
        axes[1].imshow(closed_mask.T, cmap="gray", origin="lower")
        # axes[2].imshow(closed_mask.T, cmap="gray", origin="lower")
        # plt.show()
        plt.close()
    elif dim == "3D" or dim == "2D":
        threshold = np.percentile(magn, 20)
        background_mask = magn > threshold
    else:
        raise ValueError("Dim is not correct")
    return closed_mask.astype(np.bool)
config = PreprocessConfig(temporal_std_threshold_q=600.0, magnitude_threshold_q=300)
# возвращаем (y, x, t)
phase, magn, metadata = load_dicom(patient_path)
mask = make_background_mask(phase, magn, metadata['Type'], config.temporal_std_threshold_q, config.magnitude_threshold_q)
Corrector = BackGroundPhaseCorrector(phase, mask)
Corrector.fit()
Corrector.correct(iterations=1)
# Corrector.predict()
Corrector.compute_qc(view=True)

from collections.abc import Callable

from dataclasses import dataclass
from patient import Patient
import numpy as np
import matplotlib.pyplot as plt

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
    """Класс, реализующий коррекцию фазового дрейфа на фоновых пикселях
    Каждый слайс фитится отедельно, затем происходит сглаживание параметров моделей во времени
    При этом для фита используется маска фона самого первого слайса"""
    def __init__(self, phase: np.ndarray, mask: np.ndarray, poly_order: int =2, model: str ='polynom', robust: bool =True, qc: bool=True):
        self.robust_used = None
        self.initial_model = None
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

    def fit(self) -> tuple[Callable, np.ndarray]:
        """Делает фит модели коррекции фона"""

        # TODO: добавить параметры фита в аргументы
        # coef = ['a','b','c','d','e','f','g','h','k','m','n']
        if self.model_type == 'polynom':
            if self.poly_order == 2:
                def curve(x, a, b, c, d, e, f):
                    return a + b*x[0] + c*x[1] + d*x[0]*x[1] + e * x[0] * x[0] + f * x[1] * x[1]
            else:
                raise NotImplementedError("This polyorder is currently unsupported")
        else:
            raise ValueError("Incorrect model type")

        if self.fit_mask is None:
            mask = self.mask
        else:
            mask = self.fit_mask

        self.background_coords = np.argwhere(mask == 1)

        #
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

        if self.fit_mask is None:
            self.initial_model = self.model

        return self.model

    def predict(self, robust_used: bool = False) -> np.ndarray:
        #unpack model
        if robust_used:

            fit_func, params = self.model
            print("Predicting...")

            corrected_phase = np.ndarray(shape=(self.shape[2], self.shape[0], self.shape[1]))
            # предикт фазы для всего объема TODO: вынести предикт в отдельную функцию
            for t in range(self.shape[2]):
                time_slice = self.phase[:,:,t]
                # TODO: попробовать реализовать строчку ниже с помощью np.vectorize
                predicted = np.asarray([[fit_func((i,j), *params[t]) for j in range(self.shape[1])] for i in range(self.shape[0])])
                corrected = time_slice - predicted
                corrected_phase[t] = corrected
            print("Done\n")
        else:
            fit_func, params = self.initial_model
            print("Predicting...")

            corrected_phase = np.ndarray(shape=(self.shape[2], self.shape[0], self.shape[1]))
            # предикт фазы для всего объема
            for t in range(self.shape[2]):
                time_slice = self.phase[:, :, t]

                predicted = np.asarray(
                    [[fit_func((i, j), *params[t]) for j in range(self.shape[1])] for i in range(self.shape[0])])
                corrected = time_slice - predicted
                corrected_phase[t] = corrected
            print("Done\n")

        return corrected_phase

    def correct(self, iterations: int = 3):
        print("Correcting the fit...")
        if self.fit_mask is None:
            self.fit_mask = self.mask.copy()
        bad_count = 0
        for i in range(iterations):
            fit_func, params0 = self.model

            background_slice = self.phase[:,:,0][self.fit_mask]
            # background_slice[~self.mask] = np.nan # ?
            predicted = np.asarray([fit_func(xy, *params0[0]) for xy in self.background_coords])
            residual = background_slice - predicted

            # TODO: need to be upgraded
            sigma = 1.4826 * np.median(abs(residual - np.median(residual)))
            # sigma = np.std(residual,ddof=1)
            # res_mask = abs(residual) > 3*sigma
            bad_idx = np.argwhere(abs(residual) > 4*sigma)

            if len(np.argwhere(self.fit_mask)) <= 0.85 * len(np.argwhere(self.mask)):
                break

            bad_count_n = len(bad_idx)
            if np.isclose(bad_count_n, bad_count, atol=10):
                break
            else:
                bad_count = bad_count_n

            for idx in bad_idx:

                self.fit_mask[*self.background_coords[*idx]] = 0
                    # self.background_coords.pop(idx)
            self.fit()
        self.robust_used = True
        print("Done\n")

    def compute_qc(self, view: bool = False):
        print("QC:\n")
        corrected_raw = self.predict(robust_used=False)


        # old_slice = self.phase[:,:,0][self.mask]
        # new_slice = self.phase[:,:, 0][self.fit_mask]
        raw_mean = 0
        raw_std = 0
        old_background_mean = 0
        old_background_std = 0

        for t in range(self.shape[2]):
            raw_phase = self.phase[:, :, t][self.mask]
            raw_mean += np.mean(raw_phase)
            raw_std += np.std(raw_phase, ddof=1)

            old_slice = corrected_raw[t][self.mask]
            old_background_mean += np.mean(old_slice)
            old_background_std += np.std(old_slice, ddof=1)

        raw_mean /= self.shape[2]
        raw_std /= self.shape[2]
        old_background_mean /= self.shape[2]
        old_background_std /= self.shape[2]

        print(f"Mean in the background before/after phase correction:  {raw_mean:.3f} / {old_background_mean:.3f}")
        print(f"Std in the background before/after phase correction:  {raw_std:.3f} / {old_background_std:.3f}\n")

        if self.robust_used:
            corrected_new = self.predict(robust_used=self.robust_used)

            new_background_mean = 0
            new_background_std = 0

            for t in range(self.shape[2]):
                new_slice = corrected_new[t][self.fit_mask]
                new_background_mean += np.mean(new_slice)
                new_background_std += np.std(new_slice, ddof=1)

            new_background_mean /= self.shape[2]
            new_background_std /= self.shape[2]

            print(f"Mean in the background before/after robust correction: {old_background_mean:.3f} / {new_background_mean:.3f}")
            print(f"Std in the background before/after robust correction: {old_background_std:.3f} / {new_background_std:.3f}\n")

        if view:
            if self.robust_used:
                fig, ax = plt.subplots(2,2, figsize=(12,12), sharex=False)
                ax[0, 0].imshow(self.mask)
                ax[0, 1].imshow(self.fit_mask)
                ax[1, 0].imshow(self.phase[:,:,0], cmap="gray", origin="lower")
                ax[1, 1].imshow(corrected_new[0], cmap="gray", origin="lower")
                plt.show()
                plt.close(fig)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=True)
                ax.imshow(self.mask)
                plt.show()
                plt.close(fig)
                fig2, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
                ax[0].imshow(self.phase[:,:,0], cmap="gray", origin="lower")
                ax[1].imshow(corrected_raw[0], cmap="gray", origin="lower")
                plt.show()
                plt.close(fig2)
        print("QC end")

class AliasingCorrector:
    def __init__(self, patient: Patient):
        self.jump_value = patient.metadata['jump_value']
        self.shape = patient.corrected_phase.shape
        self.phase = patient.corrected_phase
        self.aliasing_mask = np.ndarray(self.shape)

    def temporal_unwrap(self, view: bool = False) -> tuple[np.ndarray, np.ndarray]:
        print("Performing temporal unwrapping...")
        unwrapped = np.ndarray(self.shape)

        for t in range(1, self.shape[0]):
            slice_t1 = self.phase[t]
            slice_t0 = self.phase[t-1]
            jump = slice_t1-slice_t0
            jump_mask = (abs(jump) > self.jump_value)
            new_slice_t1 = np.ndarray(slice_t1.shape)
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    if jump[i,j] > self.jump_value:
                        new_slice_t1[i,j] = slice_t1[i,j]-2*self.jump_value
                    elif jump[i,j] < -self.jump_value:
                        new_slice_t1[i,j] = slice_t1[i,j]+2*self.jump_value
                    else:
                        new_slice_t1[i,j] = slice_t1[i,j]
            unwrapped[t] = new_slice_t1
            self.aliasing_mask[t] = jump_mask

        slice_t0 = self.phase[-1]
        slice_t1 = self.phase[0]
        jump = slice_t1 - slice_t0
        jump_mask = (abs(jump) > self.jump_value)
        new_slice_t1 = np.ndarray(slice_t1.shape)
        for i in range(self.shape[1]):
            for j in range(self.shape[2]):
                if jump[i, j] > self.jump_value:
                    new_slice_t1[i, j] = slice_t1[i, j] - 2 * self.jump_value
                elif jump[i, j] < -self.jump_value:
                    new_slice_t1[i, j] = slice_t1[i, j] + 2 * self.jump_value
                else:
                    new_slice_t1[i,j] = slice_t1[i,j]
        unwrapped[0] = slice_t1
        self.aliasing_mask[0] = jump_mask
        if view:
            for t in range(self.shape[0]):
                fig, ax = plt.subplots(3, 1, sharex=True, figsize = (16,16))
                ax[0].imshow(unwrapped[t], cmap="gray", origin="lower")
                ax[1].imshow(self.phase[t], cmap="gray", origin="lower")
                ax[2].imshow(self.aliasing_mask[t], cmap="gray", origin="lower")
                plt.show()
                plt.close(fig)
        return unwrapped, self.aliasing_mask

def is_raw_image(phase, venc):
    return np.max(phase) > venc




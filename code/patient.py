import re
from pathlib import Path
import json
import nibabel as nib
import os
import pydicom
from dcm2niix import main
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Patient:
    def __init__(self, path):
        self.path_to_dicom_dir = path
        self.dicom_phase = None
        self.magn = None
        self.raw_phase = None #(y,x,t)
        self.corrected_phase = None #(t, y, x) after BackgroundCorrection and AliasingCorrection

        self.metadata = None
        self.image_dimension = None
        self.background_mask = None

    def load_dicom(self) -> tuple:

        """Загрузка dicom папки, возврат phase, magnitude, metadata dict из настроек"""

        if not any(Path(self.path_to_dicom_dir).glob("*ph.nii*")):
            print("Converting dicom to Nifti")
            try:
                exit_code = main(
                    ["-z", "y", "-v", "1", "-m", "y", '-o', self.path_to_dicom_dir, self.path_to_dicom_dir]
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
            print(status)
        phase_img = None
        magn_img = None
        VENC = None
        for file in Path(self.path_to_dicom_dir).glob('*ph.nii*'):
            phase_img = nib.load(os.path.join(self.path_to_dicom_dir, file))
        for file in Path(self.path_to_dicom_dir).glob('*.json'):
            with open(file, 'r') as f:
                metadata = json.load(f)
            if "MAG" in metadata["SeriesDescription"]:
                magn_img = nib.load(os.path.join(self.path_to_dicom_dir, file.with_suffix('.nii.gz')))
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
            self.image_dimension = "2D"
        elif len(shape) == 3:
            if shape[2] == 1:
                self.image_dimension = "2D"
            else:
                self.image_dimension = "3D"
        elif len(shape) == 4:
            if shape[3] <= 3:
                self.image_dimension = "3D"
                phase_img_data = phase_img_data[:, :, :, 1]
                magn_img_data = magn_img_data[:, :, :, 1]
            elif shape[2] == 1:
                self.image_dimension = "2D+t"
                phase_img_data = phase_img_data[:, :, 0, :]
                magn_img_data = magn_img_data[:, :, 0, :]
            else:
                self.image_dimension = "4D"
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
        phase_for_hist = phase_slice.ravel()
        axes[1, 0].hist(magn_for_hist, density=True, bins=24)
        # axes[1,1].hist(phase_for_hist, density=False, bins=64)
        # plt.show()
        plt.savefig("test.jpg")
        plt.close(fig)
        plt.figure(figsize=(12, 12))
        plt.hist(phase_for_hist, bins=24)
        plt.savefig("phase_hist.jpg")
        plt.close()

        self.magn = magn_img_data
        self.raw_phase = phase_img_data
        self.metadata = img_data
        self.metadata['jump_value'] = 4096
        return phase_img_data, magn_img_data, img_data

    def make_background_mask(self, std_thres, magn_thres):
        if self.image_dimension == "4D" or self.image_dimension == "2D+t":
            # Берем дисперсию фазы и первый фрейм магнитуды
            magn_slice = self.magn[:, :, 0]
            std = np.std(self.raw_phase, axis=2, ddof=1)
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
        elif self.image_dimension == "3D" or self.image_dimension == "2D":
            threshold = np.percentile(self.magn, 20)
            background_mask = self.magn > threshold
        else:
            raise ValueError("Dim is not correct")
        self.background_mask = closed_mask.astype(np.bool)
        return closed_mask.astype(np.bool)

    @property
    def velocity_map(self):
        return self.corrected_phase * self.metadata['VENC'] / self.metadata['jump_value']
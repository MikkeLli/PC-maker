import numpy
import nibabel as nib
import dicom2nifti
import dcm2niix


# Converts all DICOM series in the folder to separate NIfTI files


"""Этот файл осуществляет преообработку данных PC-MRI:
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
def load_dicom(dicom_dir):
    # TODO: загрузка dicom, опционально VENC, и тд, мб с json
    dicom2nifti.convert_directory(dicom_dir, dicom_dir, compression=True, reorient=True)
    return
def background_phase_correction():
    # TODO: сделать непосредственно коррекцию
    return
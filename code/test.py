from pydicom import *

ds = dcmread(r"/Patients/Aqueduct/37_flow_25_aqueduct_P/flow_25_aqueduct_P_i1.dcm")

print(ds.Manufacturer)
print(ds.ManufacturerModelName)

print(ds.RescaleSlope)
print(ds.RescaleIntercept)

print(ds.BitsStored)
print(ds.HighBit)
print(ds.PixelRepresentation)

print(ds[(0x0018,0x9197)])   # Velocity Encoding Sequence, если есть
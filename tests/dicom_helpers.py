"""Shared in-memory DICOM builder for the CT-ingestion tests."""

import io

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def dicom_bytes(
    instance_number,
    fill,
    *,
    slope=1.0,
    intercept=-1024.0,
    rows=4,
    cols=4,
    series_uid=None,
    frames=1,
):
    """Build a minimal uncompressed CT DICOM slice as an in-memory BytesIO.

    Defaults produce a single-frame grayscale slice. Pass ``series_uid`` to place
    the slice in a specific series, or ``frames`` > 1 for a multi-frame volume —
    both exercise the ``load_ct_volume`` validation paths.
    """
    ds = Dataset()
    ds.InstanceNumber = instance_number
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    if series_uid is not None:
        ds.SeriesInstanceUID = series_uid
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    shape = (frames, rows, cols) if frames > 1 else (rows, cols)
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.PixelData = np.full(shape, fill, dtype=np.int16).tobytes()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = fm
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    buf.seek(0)
    return buf

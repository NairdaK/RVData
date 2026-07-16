"""
CARMENES Level 2 reader.

This module contains the CARMENES-to-RVData Level 2 translator scaffold.  The
base model already creates the required RVData L2 extensions; the methods below
are organized around filling those extensions from native CARMENES products.
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from rvdata.core.models.level2 import RV2


class CARMENESRV2(RV2):
    """
    Read CARMENES data products and convert them into RVData Level 2 format.

    This class is intentionally structured as a translator skeleton.  The
    public entry point is :meth:`_read`, which is called by
    ``RV2.from_fits(..., instrument="CARMENES")`` once the instrument is
    registered.  Fill in the instrument-specific methods below as the native
    CARMENES product layout is mapped onto the RVData L2 extensions.

    Expected implementation areas
    -----------------------------
    - ``_trace_spec``: define native flux/wavelength/variance/blaze HDU names
      for the science trace.
    - ``_populate_barycentric_extensions``: fill ``BARYCORR_KMS``,
      ``BARYCORR_Z``, and ``BJD_TDB``.
    - ``_populate_primary_header``: map the CARMENES primary header to the
      standardized RVData primary header.
    - Optional metadata methods for exposure meter, telemetry, DRP config,
      receipt, drift, tellurics, sky models, and extension descriptions.

    Example
    -------
    >>> from rvdata.instruments.carmenes.level2 import CARMENESRV2
    >>> rv2 = CARMENESRV2.from_fits("carmenes_file.fits", instrument="CARMENES")
    >>> rv2.to_fits("carmenes_L2_standard.fits")
    """

    instrument_name = "CARMENES"

    def _read(self, hdul1: fits.HDUList, **kwargs) -> None:
        """
        Populate this RVData Level 2 object from a native CARMENES FITS file.

        Parameters
        ----------
        hdul1 : fits.HDUList
            Open CARMENES FITS HDU list passed in by ``RVDataModel.read``.
        **kwargs
            Reserved for auxiliary files or conversion options.
        """


        self._validate_input(hdul1, **kwargs)
        self._populate_instrument_header(hdul1)
        self._populate_trace_extensions(hdul1, **kwargs)
        self._populate_order_table()
        self._populate_barycentric_extensions(hdul1, **kwargs)

        l0file = kwargs.get("l0file")  # here the raw file can be provided if needed (to be decided)

        if l0file is not None:
            with fits.open(l0file, memmap=False) as hdul0:
                self._populate_optional_extensions(hdul1, hdul0=hdul0, **kwargs)
        else:
            self._populate_optional_extensions(hdul1, hdul0=None, **kwargs)
            
        self._populate_primary_header(hdul1, **kwargs)
        self._populate_extension_descriptions()

    # ------------------------------------------------------------------
    # High-level translation steps

    def _validate_input(self, hdul1: fits.HDUList, **kwargs) -> None:
        """Validate that the input looks like a CARMENES product, and find out, whether it is a VIS or a NIR file."""

        if "PRIMARY" not in hdul1:
            raise ValueError("CARMENES input must contain a PRIMARY HDU.")
        
        self.channel = str(hdul1["PRIMARY"].header.get("SUBSYS", "")).lower()
        if self.channel not in ("vis", "nir"):
            raise ValueError("CARMENES channel must be either 'vis' or 'nir'; "
                f"got {self.channel!r}."
        
        # One should probably provide here the info on, which fiber is being provided (A or B, sci or cal)
        # for now fiber A (sci)
        self.trace_type = 'sci'
    )
        

    def _populate_instrument_header(self, hdul1: fits.HDUList) -> None:
        """Store the native primary header as ``INSTRUMENT_HEADER``."""

        self.set_header("INSTRUMENT_HEADER", OrderedDict(hdul1["PRIMARY"].header))

    def _populate_trace_extensions(self, hdul1: fits.HDUList, **kwargs) -> None:
        """Populate ``TRACE1_FLUX/WAVE/VAR/BLAZE`` image extensions."""

        trace_spec = self._trace_spec(hdul1, **kwargs)
        out_prefix = "TRACE1_"

        flux_data, flux_header = self._read_image_hdu(hdul1, trace_spec["flux"])
        wave_data, wave_header = self._read_image_hdu(hdul1, trace_spec["wave"])
        var_data, var_header = self._read_image_hdu(hdul1, trace_spec["var"])

        blaze_ext = trace_spec.get("blaze")
        if blaze_ext is None:
            blaze_data = np.ones_like(flux_data, dtype=float)
            blaze_header = fits.Header()
            blaze_header["BLZNORM"] = (True, "Blaze is normalized")
            blaze_header["BLAZESRC"] = (
                "NONE",
                "No native blaze provided; array set to unity",
            )
        else:
            blaze_data, blaze_header = self._read_image_hdu(hdul1, blaze_ext)

        self._set_or_create_image(out_prefix + "FLUX", flux_data, flux_header)
        self._set_or_create_image(out_prefix + "WAVE", wave_data, wave_header)
        self._set_or_create_image(out_prefix + "VAR", var_data, var_header)
        self._set_or_create_image(out_prefix + "BLAZE", blaze_data, blaze_header)

    def _populate_order_table(self, wave_ext: str = "TRACE1_WAVE") -> None:
        """Build ``ORDER_TABLE`` from a populated wavelength extension."""

        if wave_ext not in self.data or self.data[wave_ext].size == 0:
            raise NotImplementedError(
                "Populate trace wavelength data before building ORDER_TABLE, "
                "or override _populate_order_table for CARMENES."
            )

        wavelengths = np.asarray(self.data[wave_ext])
        order_table = pd.DataFrame(
            {
                "ECHELLE_ORDER": self._echelle_orders(wavelengths),
                "ORDER_INDEX": np.arange(wavelengths.shape[0]),
                "WAVE_START": np.nanmin(wavelengths, axis=1),
                "WAVE_END": np.nanmax(wavelengths, axis=1),
            }
        )
        self.set_data("ORDER_TABLE", order_table)

    def _populate_barycentric_extensions(
        self, hdul1: fits.HDUList, **kwargs
    ) -> None:
        """Populate barycentric correction and BJD extensions."""

        raise NotImplementedError(
            "Map CARMENES barycentric correction data to BARYCORR_KMS, "
            "BARYCORR_Z, and BJD_TDB."
        )

    def _populate_optional_extensions(self, hdul1: fits.HDUList, **kwargs) -> None:
        """
        Populate optional RVData L2 extensions when CARMENES products provide them.

        Fill this method with instrument-specific handling for extensions such
        as ``EXPMETER``, ``TELEMETRY``, ``DRP_CONFIG``, ``RECEIPT``,
        ``TRACEi_DRIFT``, ``TRACEi_TELLURIC``, and ``TRACEi_SKY``.
        """

    def _populate_primary_header(self, hdul1: fits.HDUList, **kwargs) -> None:
        """Populate the standardized RVData primary header."""

        raise NotImplementedError(
            "Map the native CARMENES primary header to the RVData PRIMARY header."
        )

    def _populate_extension_descriptions(self) -> None:
        """
        Populate ``EXT_DESCRIPT``.

        If ``config/ext_descript.csv`` exists next to this module, it is used.
        Otherwise a compact table is generated from currently present
        extensions so interim products remain inspectable while the translator
        is under development.
        """

        ext_path = os.path.join(os.path.dirname(__file__), "config", "ext_descript.csv")
        if os.path.exists(ext_path):
            ext_descript = pd.read_csv(ext_path)
            if "Comments" in ext_descript.columns:
                ext_descript = ext_descript.drop(columns=["Comments"])
            ext_descript = ext_descript[
                ext_descript["Name"].isin(self.extensions.keys())
            ].copy()
        else:
            ext_descript = pd.DataFrame(
                {
                    "Name": list(self.extensions.keys()),
                    "Description": [
                        "CARMENES translator scaffold placeholder"
                        for _ in self.extensions
                    ],
                }
            )

        self.set_data("EXT_DESCRIPT", ext_descript.reset_index(drop=True))

    # ------------------------------------------------------------------
    # Instrument-specific hooks to fill in

    def _trace_spec(self, hdul1: fits.HDUList, **kwargs) -> dict[str, object]:
        """
        Return native HDU mappings for the science trace.

        The returned dictionary should contain:

        - ``flux``: native flux HDU name.
        - ``wave``: native wavelength HDU name.
        - ``var``: native variance HDU name.
        - ``blaze``: optional native blaze HDU name.  Use ``None`` to create a
          temporary all-ones blaze array.

        Example
        -------
        return {
            "flux": "SCI_FLUX",
            "wave": "SCI_WAVE",
            "var": "SCI_VAR",
            "blaze": "SCI_BLAZE",
        }
        """


        return {
            "flux": "SPEC",
            "wave": "WAVE",
            "var": "SIG",
            "blaze": None,
        }


        

    def _echelle_orders(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return physical echelle orders for the wavelength array rows."""

        return np.arange(wavelengths.shape[0])

    # ------------------------------------------------------------------
    # Small helpers

    @staticmethod
    def _require_hdu(hdul1: fits.HDUList, ext_name: str):
        """Return an HDU or raise a helpful error if it is missing."""

        if ext_name not in hdul1:
            raise KeyError(f"Required CARMENES HDU '{ext_name}' not found.")
        return hdul1[ext_name]

    def _read_image_hdu(
        self, hdul1: fits.HDUList, ext_name: str
    ) -> tuple[np.ndarray, fits.Header]:
        """Read an image HDU's data and header."""

        hdu = self._require_hdu(hdul1, ext_name)
        if hdu.data is None:
            raise ValueError(f"CARMENES HDU '{ext_name}' has no data.")
        return np.asarray(hdu.data), hdu.header

    def _set_or_create_image(
        self, ext_name: str, data: np.ndarray, header: fits.Header | OrderedDict
    ) -> None:
        """Set an existing image extension or create it if this trace is new."""

        if ext_name in self.extensions:
            self.set_header(ext_name, OrderedDict(header))
            self.set_data(ext_name, data)
        else:
            self.create_extension(
                ext_name, "ImageHDU", data=data, header=OrderedDict(header)
            )

    def _set_or_create_table(
        self,
        ext_name: str,
        data: Table | pd.DataFrame,
        header: fits.Header | OrderedDict | None = None,
    ) -> None:
        """Set an existing table extension or create it if absent."""

        if ext_name in self.extensions:
            if header is not None:
                self.set_header(ext_name, OrderedDict(header))
            self.set_data(ext_name, data)
        else:
            self.create_extension(
                ext_name,
                "BinTableHDU",
                data=data,
                header=OrderedDict(header or {}),
            )

"""Optional presentation image export for DuoStar Basic (does not affect measurement)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from scipy.ndimage import map_coordinates


def create_annotated_image(data, wcs, row, x_a, y_a, x_b, y_b,
                           output_path: Path, field_arcmin: float = 5.0) -> Path:
    """Save a 5x5 arcminute north-up/east-left annotated PNG from measured centroids.

    Raises ValueError when a complete field is not present; leaves the measurement intact.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from astropy.wcs.utils import pixel_to_skycoord

    if not np.all(np.isfinite([x_a, y_a, x_b, y_b])):
        raise ValueError('Invalid measured component positions.')
    ca = pixel_to_skycoord(x_a, y_a, wcs, origin=0, mode='all')
    cb = pixel_to_skycoord(x_b, y_b, wcs, origin=0, mode='all')
    # The spherical midpoint avoids assumptions about the CCD orientation.
    centre = ca.directional_offset_by(ca.position_angle(cb), ca.separation(cb) / 2)
    scales = np.abs(proj_plane_pixel_scales(wcs.celestial)) * 3600.0
    scale = float(np.min(scales))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid FITS WCS scale.')
    n = int(np.clip(math.ceil(field_arcmin * 60 / scale), 200, 1000))
    side = field_arcmin * 60.0
    yy, xx = np.mgrid[0:n, 0:n]
    # Image coordinates: right=west, up=north. Offsets are precise locally.
    west = (xx - (n-1)/2) * side/n
    north = (yy - (n-1)/2) * side/n
    coords = centre.spherical_offsets_by((-west)*u.arcsec, north*u.arcsec)
    px, py = skycoord_to_pixel(coords, wcs, origin=0, mode='all')
    h, w = data.shape
    if (np.any(~np.isfinite(px)) or np.any(~np.isfinite(py)) or
            px.min() < 0 or py.min() < 0 or px.max() > w-1 or py.max() > h-1):
        raise ValueError('The full 5 arcminute field extends outside the FITS image.')
    vals = map_coordinates(data.astype(float), [py, px], order=1, mode='nearest')
    finite = vals[np.isfinite(vals)]
    if not len(finite):
        raise ValueError('No valid pixels in annotation crop.')
    lo, hi = np.percentile(finite, [8, 99.7])
    if hi <= lo:
        hi = lo + 1
    normalized = np.arcsinh(np.clip((vals - lo)/(hi-lo), 0, None)*8) / np.arcsinh(8)
    normalized = np.clip(normalized, 0, 1)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    fig.patch.set_facecolor('white')
    ax.imshow(normalized, origin='lower', cmap='gray_r', extent=(-side/2,side/2,-side/2,side/2),
              interpolation='nearest', vmin=0, vmax=1)

    for label, component in [('A', ca), ('B', cb)]:
        east, north_offset = centre.spherical_offsets_to(component)
        cx = -east.to_value(u.arcsec)
        cy = north_offset.to_value(u.arcsec)
        length = math.hypot(cx, cy)
        dx, dy = ((cx/length,cy/length) if length > 0 else (0,1))
        # Lines point outward from each centroid; label doesn't cover the star.
        ax.plot([cx + dx*8, cx + dx*20], [cy + dy*8, cy + dy*20],
                color='black', lw=1.2)
        t = ax.text(cx + dx*30, cy + dy*30, label, fontsize=15,
                    ha='center', va='center', color='black', family='serif')
        t.set_path_effects([effects.withStroke(linewidth=2.6, foreground='white')])

    # Fixed 60 arcsec scale in the upper right.
    sx, sy = 45.0, side/2-28.0
    ax.plot([sx,sx+60], [sy,sy], color='#ddaa00', lw=1.4)
    ax.plot([sx,sx],[sy-3,sy+3], color='#ddaa00', lw=1)
    ax.plot([sx+60,sx+60],[sy-3,sy+3], color='#ddaa00', lw=1)
    ax.text(sx+30,sy+7,'1′',ha='center',va='bottom',color='#b88d00',fontsize=12)
    # North up, East left for the reprojected image.
    ax.annotate('', xy=(-side/2+27,side/2-25),xytext=(-side/2+27,side/2-58),
                arrowprops=dict(arrowstyle='->',color='#d4a600',lw=1.2))
    ax.annotate('', xy=(-side/2+12,side/2-58),xytext=(-side/2+42,side/2-58),
                arrowprops=dict(arrowstyle='->',color='#d4a600',lw=1.2))
    ax.text(-side/2+27,side/2-18,'N',color='#b88d00',ha='center',fontsize=10)
    ax.text(-side/2+7,side/2-58,'E',color='#b88d00',ha='center',va='center',fontsize=10)
    ax.set(xlim=(-side/2,side/2),ylim=(-side/2,side/2))
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=160, facecolor='white')
    plt.close(fig)
    return output_path

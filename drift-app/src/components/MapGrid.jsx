import React, { useState } from 'react';
import { Polyline, useMap } from 'react-leaflet';

export default function MapGrid({ gridSizeKm = 1 }) {
  const map = useMap();
  const [lines, setLines] = useState([]);

  React.useEffect(() => {
    const updateGrid = () => {
      const bounds = map.getBounds();
      const centerLat = map.getCenter().lat;
      const n = bounds.getNorth();
      const s = bounds.getSouth();
      const e = bounds.getEast();
      const w = bounds.getWest();

      // 1 degree latitude is approx 111.32 km
      const latStep = gridSizeKm / 111.32;
      const lngStep = gridSizeKm / (111.32 * Math.cos(centerLat * Math.PI / 180));

      const newLines = [];

      for (let lat = Math.floor(s / latStep) * latStep; lat <= n + latStep; lat += latStep) {
        newLines.push([[lat, w - lngStep], [lat, e + lngStep]]);
      }
      for (let lng = Math.floor(w / lngStep) * lngStep; lng <= e + lngStep; lng += lngStep) {
        newLines.push([[s - latStep, lng], [n + latStep, lng]]);
      }
      setLines(newLines);
    };

    updateGrid();
    map.on('moveend', updateGrid);
    return () => {
      map.off('moveend', updateGrid);
    };
  }, [map, gridSizeKm]);

  return (
    <>
      {lines.map((pos, idx) => (
        <Polyline key={idx} positions={pos} pathOptions={{ color: '#1E5C9E', weight: 1, opacity: 0.2 }} interactive={false} />
      ))}
    </>
  );
}

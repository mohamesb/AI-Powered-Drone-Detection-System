"""
Drone swarm simulator.

Each drone:
  • flies a waypoint route over Oslo
  • broadcasts its position to the swarm every tick
  • avoids collisions by inspecting peers' positions and applying a repulsion
    vector when another drone enters its safety bubble

This is the simplest realistic model of "drones share their location to avoid
collisions" — distributed, no central planner, each agent reasons locally.

Coordinates are in lat/lon. Speeds are in metres/second. Internally we
convert short distances to a flat-earth approximation for the avoidance math,
which is fine at ~10km scales.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List

# Oslo city centre as the operational area
OSLO_LAT = 59.9139
OSLO_LON = 10.7522

# Tunables
SAFETY_RADIUS_M = 80.0      # metres — within this, drones repel each other
WAYPOINT_RADIUS_M = 30.0    # metres — counted as "reached"
NOMINAL_SPEED_MPS = 12.0    # cruise speed
AVOID_GAIN = 1.6            # strength of repulsion
TICK_SECONDS = 0.2          # simulation step


def meters_to_latlon(lat: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Approximate metres → degrees offset at a given latitude."""
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * math.cos(math.radians(lat)))
    return dlat, dlon


def latlon_distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Haversine distance in metres — accurate at all scales."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(a_lat), math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlam = math.radians(b_lon - a_lon)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class Drone:
    id: str
    lat: float
    lon: float
    altitude_m: float = 80.0
    heading_deg: float = 0.0    # 0 = north, 90 = east
    speed_mps: float = NOMINAL_SPEED_MPS
    waypoints: List[tuple[float, float]] = field(default_factory=list)
    waypoint_idx: int = 0
    camera_fov_deg: float = 60.0    # camera horizontal field of view

    def current_waypoint(self) -> tuple[float, float]:
        return self.waypoints[self.waypoint_idx % len(self.waypoints)]

    def advance_waypoint_if_reached(self):
        wp_lat, wp_lon = self.current_waypoint()
        if latlon_distance_m(self.lat, self.lon, wp_lat, wp_lon) < WAYPOINT_RADIUS_M:
            self.waypoint_idx = (self.waypoint_idx + 1) % len(self.waypoints)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "altitude_m": round(self.altitude_m, 1),
            "heading_deg": round(self.heading_deg, 1),
            "speed_mps": round(self.speed_mps, 2),
            "camera_fov_deg": self.camera_fov_deg,
        }


class Swarm:
    """Manages all drones and runs the simulation tick."""

    def __init__(self, n_drones: int = 6, seed: int | None = 42):
        self.rng = random.Random(seed)
        self.drones: List[Drone] = []
        self._spawn(n_drones)

    def _spawn(self, n: int):
        """Distribute drones around Oslo, each with a unique looping route."""
        for i in range(n):
            # Spread spawn points in a ring ~1.5 km out from city centre
            angle = (i / n) * 2 * math.pi
            radius_m = 1200 + self.rng.uniform(-200, 200)
            dlat, dlon = meters_to_latlon(
                OSLO_LAT,
                radius_m * math.cos(angle),
                radius_m * math.sin(angle),
            )
            spawn_lat = OSLO_LAT + dlat
            spawn_lon = OSLO_LON + dlon

            # Generate 3-5 waypoints in a loose loop
            waypoints = []
            n_wp = self.rng.randint(3, 5)
            for j in range(n_wp):
                a = self.rng.uniform(0, 2 * math.pi)
                r = self.rng.uniform(400, 2000)
                wp_dlat, wp_dlon = meters_to_latlon(
                    OSLO_LAT, r * math.cos(a), r * math.sin(a)
                )
                waypoints.append((OSLO_LAT + wp_dlat, OSLO_LON + wp_dlon))

            self.drones.append(Drone(
                id=f"DR-{i+1:02d}",
                lat=spawn_lat,
                lon=spawn_lon,
                altitude_m=60 + self.rng.uniform(0, 60),
                waypoints=waypoints,
            ))

    def step(self, dt: float = TICK_SECONDS):
        """Advance the whole swarm by dt seconds."""
        for d in self.drones:
            self._update_drone(d, dt)

    def _update_drone(self, d: Drone, dt: float):
        # 1. Compute desired direction toward current waypoint
        wp_lat, wp_lon = d.current_waypoint()
        # Local flat-earth offsets
        dx_to_wp = (wp_lon - d.lon) * 111_320.0 * math.cos(math.radians(d.lat))
        dy_to_wp = (wp_lat - d.lat) * 111_320.0
        wp_dist = math.hypot(dx_to_wp, dy_to_wp) or 1e-9
        desire_x = dx_to_wp / wp_dist
        desire_y = dy_to_wp / wp_dist

        # 2. Apply repulsion from any drone inside the safety bubble.
        #    This is the collision-avoidance step — purely local, uses only
        #    each peer's broadcast position.
        repel_x = repel_y = 0.0
        for other in self.drones:
            if other.id == d.id:
                continue
            dist_m = latlon_distance_m(d.lat, d.lon, other.lat, other.lon)
            if dist_m < SAFETY_RADIUS_M and dist_m > 1e-3:
                # Vector pointing away from the other drone
                ox = (d.lon - other.lon) * 111_320.0 * math.cos(math.radians(d.lat))
                oy = (d.lat - other.lat) * 111_320.0
                norm = math.hypot(ox, oy) or 1e-9
                weight = (SAFETY_RADIUS_M - dist_m) / SAFETY_RADIUS_M
                repel_x += (ox / norm) * weight
                repel_y += (oy / norm) * weight

        # 3. Combine waypoint pull + collision push
        vx = desire_x + AVOID_GAIN * repel_x
        vy = desire_y + AVOID_GAIN * repel_y
        norm = math.hypot(vx, vy) or 1e-9
        vx /= norm
        vy /= norm

        # 4. Move
        step_m = d.speed_mps * dt
        dlat, dlon = meters_to_latlon(d.lat, vx * step_m, vy * step_m)
        d.lat += dlat
        d.lon += dlon

        # 5. Update heading for the camera direction visual
        d.heading_deg = (math.degrees(math.atan2(vx, vy)) + 360) % 360

        # 6. Did we reach the waypoint?
        d.advance_waypoint_if_reached()

    def snapshot(self) -> dict:
        """Return a serialisable snapshot for the frontend."""
        return {
            "ts": time.time(),
            "drones": [d.to_dict() for d in self.drones],
            "min_pair_distance_m": self._closest_pair_distance(),
        }

    def _closest_pair_distance(self) -> float:
        """Diagnostic — smallest distance between any two drones, in metres."""
        best = float("inf")
        for i, a in enumerate(self.drones):
            for b in self.drones[i+1:]:
                d = latlon_distance_m(a.lat, a.lon, b.lat, b.lon)
                best = min(best, d)
        return round(best, 1) if best != float("inf") else 0.0

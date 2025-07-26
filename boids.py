import cv2, random
import numpy as np

class Boid:

    def __init__(self, x, y, width, height):
        self.pos = np.array([x, y], dtype = float)
        angle = random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype = float)
        self.acc = np.zeros(2)
        self.max_speed = 4
        self.max_force = 0.05
        self.perception = 50
        self.width, self.height = width, height

    def edges(self, margin = 50, strength = 0.5):
        steer = np.zeros(2)
        if self.pos[0] < margin:
            steer[0] = strength
        elif self.pos[0] > self.width - margin:
            steer[0] = -strength

        if self.pos[1] < margin:
            steer[1] = strength
        elif self.pos[1] > self.height - margin:
            steer[1] = -strength

        if np.linalg.norm(steer) > 0:
            steer = self._set_mag(steer, self.max_speed)
            steer -= self.vel
            steer = self._limit(steer, self.max_force * 2)

        return steer
                        
    def update(self):
        self.vel += self.acc
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
        self.pos += self.vel
        self.acc *= 0

    def apply_force(self, force):
        self.acc += force

    def align(self, boids):
        steer = np.zeros(2)
        tot, avg_vector = 0, np.zeros(2)

        for other in boids:
            if other is self: continue # Omit current boid from calculation
            dist = np.linalg.norm(self.pos - other.pos)
            if dist < self.perception:
                avg_vector += other.vel
                tot += 1

            
        if tot > 0:
            avg_vector /= tot
            avg_vector = self._set_mag(avg_vector, self.max_speed)
            steer = avg_vector - self.vel
            steer = self._limit(steer, self.max_force)

        return steer
    
    def cohesion(self, boids):
        steer = np.zeros(2)
        tot, com = 0, np.zeros(2)

        for other in boids:
            if other is self: continue # Omit current boid from calculations
            dist = np.linalg.norm(self.pos - other.pos)
            if dist < self.perception:
                com += other.pos
                tot += 1

        if tot > 0:
            com /= tot 
            desired = com - self.pos
            desired = self._set_mag(desired, self.max_speed)
            steer = desired - self.vel
            steer = self._limit(steer, self.max_force)

        return steer
    
    def separation(self, boids):
        steer = np.zeros(2)
        tot = 0
        
        for other in boids:
            if other is self: continue # Omit current boid from calculations
            dist = np.linalg.norm(self.pos - other.pos)
            if dist < self.perception / 2:
                diff = self.pos - other.pos
                if dist != 0: diff /= dist
                steer += diff
                tot += 1

        if tot > 0:
            steer /= tot
            steer = self._set_mag(steer, self.max_speed)
            steer -= self.vel
            steer = self._limit(steer, self.max_force)

        return steer
    
    def attract(self, point, radius = 200, strength = 0.3):
        direction = point - self.pos
        dist = np.linalg.norm(direction)

        if dist < radius and dist != 0:
            force  = (direction / dist) * strength * (dist / radius)
            self.apply_force(force)
    
    def flock(self, boids, align_w = 10, coh_w = 1.0, sep_w = 1.0):
        alignment = self.align(boids) * align_w
        cohesion = self.cohesion(boids) * coh_w
        separation = self.separation(boids) * sep_w
        edges = self.edges(margin = 100)

        self.apply_force(alignment)
        self.apply_force(cohesion)
        self.apply_force(separation)
        self.apply_force(edges)

    def _set_mag(self, vec, mag):
        norm = np.linalg.norm(vec)
        if norm == 0: return vec
        return (vec / norm) * mag
    
    def _limit(self, vec, max_vel):
        norm = np.linalg.norm(vec)
        if norm > max_vel: return (vec / norm) * max_vel
        return vec
    
    def draw(self, frame):
        x, y = self.pos.astype(int)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
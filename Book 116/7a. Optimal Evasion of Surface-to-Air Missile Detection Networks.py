import math
import heapq
from collections import defaultdict

EPS = 1e-9


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(center, p):
    return math.atan2(p[1] - center[1], p[0] - center[0])


def normalize(a):
    while a < 0:
        a += 2 * math.pi
    while a >= 2 * math.pi:
        a -= 2 * math.pi
    return a


# ---------------------------------------------------------
# Circle / Segment intersection
# ---------------------------------------------------------

def segment_intersects_circle(a, b, circle):
    """
    Returns True if segment AB enters the interior
    of the circle.
    """
    cx, cy, r = circle

    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    length2 = dx * dx + dy * dy

    if length2 < EPS:
        return dist(a, (cx, cy)) < r - EPS

    t = ((cx - ax) * dx + (cy - ay) * dy) / length2
    t = max(0.0, min(1.0, t))

    px = ax + t * dx
    py = ay + t * dy

    d = math.hypot(px - cx, py - cy)

    return d < r - 1e-7


def visible_segment(a, b, circles):
    """
    Segment AB must not enter any circle interior.
    """
    for c in circles:
        if segment_intersects_circle(a, b, c):
            return False
    return True


# ---------------------------------------------------------
# Tangents from point to circle
# ---------------------------------------------------------

def tangents_point_circle(point, circle):
    """
    Returns tangent points on circle from external point.
    """
    px, py = point
    cx, cy, r = circle

    dx = px - cx
    dy = py - cy

    d2 = dx * dx + dy * dy
    d = math.sqrt(d2)

    if d <= r + EPS:
        return []

    alpha = math.atan2(dy, dx)
    beta = math.acos(r / d)

    pts = []

    for sign in (-1, 1):
        theta = alpha + sign * beta
        tx = cx + r * math.cos(theta)
        ty = cy + r * math.sin(theta)
        pts.append((tx, ty))

    return pts


# ---------------------------------------------------------
# Circle-circle tangents
# ---------------------------------------------------------

def circle_circle_tangents(c1, c2):
    """
    Returns tangent point pairs.

    Each item:
        ((x1,y1),(x2,y2))
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2

    dx = x2 - x1
    dy = y2 - y1

    d2 = dx * dx + dy * dy

    if d2 < EPS:
        return []

    res = []

    for s in (-1, 1):

        r = r1 - s * r2

        h2 = d2 - r * r

        if h2 < -EPS:
            continue

        h2 = max(h2, 0.0)

        for sign in (-1, 1):

            vx = (dx * r - sign * dy * math.sqrt(h2)) / d2
            vy = (dy * r + sign * dx * math.sqrt(h2)) / d2

            p1 = (
                x1 + r1 * vx,
                y1 + r1 * vy
            )

            p2 = (
                x2 + s * r2 * vx,
                y2 + s * r2 * vy
            )

            res.append((p1, p2))

    return res


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(list)

    def add_node(self, p):
        idx = len(self.nodes)
        self.nodes.append(p)
        return idx

    def add_edge(self, u, v, w):
        self.edges[u].append((v, w))
        self.edges[v].append((u, w))


# ---------------------------------------------------------
# Main planner
# ---------------------------------------------------------

def shortest_safe_path(start, goal, circles):

    g = Graph()

    node_id = {}
    circle_nodes = defaultdict(list)

    def get_node(p):
        key = (round(p[0], 10), round(p[1], 10))
        if key not in node_id:
            node_id[key] = g.add_node(p)
        return node_id[key]

    start_id = get_node(start)
    goal_id = get_node(goal)

    # -----------------------------------------------------
    # Tangents from start / goal
    # -----------------------------------------------------

    for ci, c in enumerate(circles):

        for p in tangents_point_circle(start, c):
            nid = get_node(p)
            circle_nodes[ci].append(nid)

        for p in tangents_point_circle(goal, c):
            nid = get_node(p)
            circle_nodes[ci].append(nid)

    # -----------------------------------------------------
    # Circle-circle tangents
    # -----------------------------------------------------

    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):

            tangents = circle_circle_tangents(
                circles[i],
                circles[j]
            )

            for p1, p2 in tangents:

                n1 = get_node(p1)
                n2 = get_node(p2)

                circle_nodes[i].append(n1)
                circle_nodes[j].append(n2)

    # -----------------------------------------------------
    # Straight visibility edges
    # -----------------------------------------------------

    N = len(g.nodes)

    for i in range(N):
        for j in range(i + 1, N):

            a = g.nodes[i]
            b = g.nodes[j]

            if visible_segment(a, b, circles):
                g.add_edge(i, j, dist(a, b))

    # -----------------------------------------------------
    # Arc edges
    # -----------------------------------------------------

    for ci, c in enumerate(circles):

        cx, cy, r = c

        unique = list(set(circle_nodes[ci]))

        if len(unique) < 2:
            continue

        angs = []

        for nid in unique:
            p = g.nodes[nid]
            angs.append(
                (normalize(angle((cx, cy), p)), nid)
            )

        angs.sort()

        m = len(angs)

        for k in range(m):

            a1, n1 = angs[k]
            a2, n2 = angs[(k + 1) % m]

            delta = a2 - a1
            if delta < 0:
                delta += 2 * math.pi

            arc_len = r * delta

            g.add_edge(n1, n2, arc_len)

    # -----------------------------------------------------
    # Dijkstra
    # -----------------------------------------------------

    pq = [(0.0, start_id)]

    best = {start_id: 0.0}
    parent = {start_id: None}

    while pq:

        d, u = heapq.heappop(pq)

        if d != best[u]:
            continue

        if u == goal_id:
            break

        for v, w in g.edges[u]:

            nd = d + w

            if nd < best.get(v, float("inf")):

                best[v] = nd
                parent[v] = u

                heapq.heappush(
                    pq,
                    (nd, v)
                )

    if goal_id not in best:
        return None

    path_nodes = []

    cur = goal_id

    while cur is not None:
        path_nodes.append(cur)
        cur = parent[cur]

    path_nodes.reverse()

    return [g.nodes[i] for i in path_nodes]


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":

    start = (0.0, 0.0)
    goal = (20.0, 0.0)

    circles = [
        (8.0, 0.0, 3.0),
        (13.0, 2.0, 2.0),
    ]

    path = shortest_safe_path(
        start,
        goal,
        circles
    )

    if path is None:
        print("No safe path exists.")
    else:
        print("Safe path:")
        for p in path:
            print(f"{p[0]:.6f}, {p[1]:.6f}")

"""Build the SUMO road network: an R x C grid, 100 m between junctions, 2 lanes per direction,
50 km/h, a fixed-time traffic light at every junction (60 s cycle, 50/50 split, 3 s amber).

    python build_network.py              # size from config.yaml (25 x 25)
    python build_network.py --rows 5     # 5 x 5 test network
Output: network/grid_<R>x<C>.net.xml
"""
import argparse
import subprocess

import sumolib

from common import binary, load_config, net_file, tag


def build(cfg):
    net, sig = cfg["network"], cfg["signal"]
    out = net_file(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    if net["rows"] != net["cols"]:
        grid_number = ["--grid.x-number", str(net["cols"]), "--grid.y-number", str(net["rows"])]
    else:
        grid_number = ["--grid.number", str(net["rows"])]
    cmd = [binary("netgenerate"), "--grid", *grid_number,
           "--grid.length", str(net["link_length_m"]), "--grid.attach-length", "0",
           "--default.lanenumber", str(net["lanes"]), "--default.speed", str(net["free_flow_speed_mps"]),
           "--default-junction-type", "traffic_light", "--no-turnarounds", "true",
           "--tls.default-type", "static", "--tls.cycle.time", str(sig["cycle_s"]),
           "--tls.yellow.time", str(sig["lost_time_s"]), "--tls.left-green.time", "0",
           "--seed", "42", "--output-file", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def check(cfg, path):
    """Every signalised junction must run NS green / amber / EW green / amber with the configured times."""
    n = sumolib.net.readNet(str(path), withPrograms=True)
    green = cfg["signal"]["cycle_s"] / 2 - cfg["signal"]["lost_time_s"]
    bad = []
    for tls in n.getTrafficLights():
        for prog in tls.getPrograms().values():
            phases = prog.getPhases()
            if len(phases) == 1:                         # corner junction: no conflicting streams
                continue
            durs = [p.duration for p in phases]
            if durs != [green, cfg["signal"]["lost_time_s"]] * 2:
                bad.append((tls.getID(), durs))
    rows, cols = cfg["network"]["rows"], cfg["network"]["cols"]
    print(f"{path.name}: {len(n.getNodes())} junctions (expected {rows * cols}), "
          f"{len(n.getEdges())} edges, {len(n.getTrafficLights())} traffic lights, "
          f"road length {n.getEdges()[0].getLength():.1f} m")
    if bad:
        raise RuntimeError(f"unexpected signal plans, e.g. {bad[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    args = ap.parse_args()
    cfg = load_config(args.rows, args.cols)
    path = build(cfg)
    check(cfg, path)
    print(f"network {tag(cfg)} -> {path}")


if __name__ == "__main__":
    main()

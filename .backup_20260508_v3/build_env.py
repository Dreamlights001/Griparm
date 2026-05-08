#!/usr/bin/env python3
"""Build a MuJoCo scene XML from URDF and inject task-specific entities."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ARM_JOINTS = ["J_jianbu", "J_dabi", "J_Upper", "J_Lower", "J_wrist", "J_hand"]
GRIPPER_JOINT = "Claw_left"


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print helper for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _normalize_urdf_mesh_paths(src_urdf: Path, out_urdf: Path) -> None:
    """Normalize mesh filename to basename so copied STL files are discoverable."""
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename:
            mesh.attrib["filename"] = Path(filename).name
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)


def _compile_urdf_to_mjcf(urdf_path: Path, meshes_dir: Path) -> ET.Element:
    """Compile URDF via MuJoCo importer and return parsed MJCF root."""
    with tempfile.TemporaryDirectory(prefix="mujoco_build_") as tmpdir:
        tmp = Path(tmpdir)
        tmp_urdf = tmp / urdf_path.name
        _normalize_urdf_mesh_paths(urdf_path, tmp_urdf)

        for stl in meshes_dir.glob("*.STL"):
            shutil.copy2(stl, tmp / stl.name)
        for stl in meshes_dir.glob("*.stl"):
            shutil.copy2(stl, tmp / stl.name)

        model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
        compiled = tmp / "compiled.xml"
        mujoco.mj_saveLastXML(str(compiled), model)
        return ET.parse(compiled).getroot()


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _set_robot_mesh_abs_paths(asset: ET.Element, meshes_dir: Path) -> None:
    for mesh in asset.findall("mesh"):
        name = mesh.attrib.get("name", "")
        if name in {"anomaly_mesh", "normal_mesh"}:
            continue
        candidate_upper = meshes_dir / f"{name}.STL"
        candidate_lower = meshes_dir / f"{name}.stl"
        if candidate_upper.exists():
            mesh.attrib["file"] = str(candidate_upper.resolve())
        elif candidate_lower.exists():
            mesh.attrib["file"] = str(candidate_lower.resolve())


def _add_assets(asset: ET.Element, anomaly_stl: Path, normal_stl: Path) -> None:
    ET.SubElement(
        asset,
        "material",
        name="normal_pine_mat",
        rgba="0.88 0.72 0.51 1",
        specular="0.25",
        shininess="0.3",
    )
    ET.SubElement(
        asset,
        "material",
        name="anomaly_pine_mat",
        rgba="0.96 0.46 0.24 1",
        specular="0.25",
        shininess="0.3",
    )
    ET.SubElement(asset, "mesh", name="anomaly_mesh", file=str(anomaly_stl.resolve()), scale="0.001 0.001 0.001")
    ET.SubElement(asset, "mesh", name="normal_mesh", file=str(normal_stl.resolve()), scale="0.001 0.001 0.001")


def _add_cameras_and_sites(worldbody: ET.Element) -> None:
    hand_body = worldbody.find(".//body[@name='Hand_Link']")
    if hand_body is None:
        raise RuntimeError("Cannot find Hand_Link in compiled MJCF.")

    ET.SubElement(
        hand_body,
        "site",
        name="tcp_site",
        type="sphere",
        size="0.004",
        pos="0.045 0.0 0.070",
        rgba="1 0.1 0.1 1",
    )

    # Active wrist camera:
    # - pose is defined in Hand_Link local frame
    # - optical direction should face +Z of Hand_Link, so rotate 180 deg around X
    ET.SubElement(
        hand_body,
        "camera",
        name="wrist",
        mode="fixed",
        pos="0 0.06 0.1",
        quat="0 1 0 0",
        fovy="58",
    )

    # Fixed global camera:
    # position = (0, 0, 1.2), lookat = (0.5, 0, 0)
    # quat computed from lookat target (w x y z).
    ET.SubElement(
        worldbody,
        "camera",
        name="global",
        mode="fixed",
        pos="0 0 1.2",
        quat="0.693375 0.138675 -0.138675 -0.693375",
        fovy="45",
    )


def _add_table_and_objects(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos="0.10 0.00 0.00",
        size="0.70 0.45 0.02",
        rgba="0.25 0.25 0.25 1",
        friction="1.0 0.005 0.0001",
    )

    anomaly = ET.SubElement(worldbody, "body", name="anomaly_0", pos="-0.40 0.10 0.02")
    ET.SubElement(anomaly, "freejoint", name="anomaly_0_free")
    ET.SubElement(
        anomaly,
        "geom",
        name="anomaly_0_geom",
        type="mesh",
        mesh="anomaly_mesh",
        material="anomaly_pine_mat",
        mass="0.06",
        friction="0.7 0.005 0.0001",
        condim="4",
    )

    y_offsets = [-0.10, -0.16, -0.22, -0.28, -0.34]
    for idx, y in enumerate(y_offsets):
        body = ET.SubElement(worldbody, "body", name=f"normal_{idx}", pos=f"{-0.50 + 0.04 * idx:.3f} {y:.3f} 0.02")
        ET.SubElement(body, "freejoint", name=f"normal_{idx}_free")
        ET.SubElement(
            body,
            "geom",
            name=f"normal_{idx}_geom",
            type="mesh",
            mesh="normal_mesh",
            material="normal_pine_mat",
            mass="0.06",
            friction="0.7 0.005 0.0001",
            condim="4",
        )


def _add_friction_to_claw_geoms(worldbody: ET.Element) -> None:
    """Elliptic cone + condim=6 + TPU-soft margin for stable grasping."""
    for geom in worldbody.findall(".//geom"):
        mesh_name = geom.attrib.get("mesh", "")
        if mesh_name in {"Claw_Link_left", "Claw_Link_right"}:
            geom.attrib["condim"] = "6"
            geom.attrib["friction"] = "1.5 0.2 0.02"   # TPU-like grip
            geom.attrib["solref"] = "0.02 1"             # softer for 2-sided convergence
            geom.attrib["solimp"] = "0.9 0.95 0.003"     # allow more penetration
            geom.attrib["margin"] = "0.0005"              # thin soft layer
        elif mesh_name in {"anomaly_mesh", "normal_mesh"}:
            geom.attrib["condim"] = "6"
            geom.attrib["friction"] = "1.0 0.05 0.005"
            geom.attrib["solref"] = "0.01 1"
            geom.attrib["solimp"] = "0.9 0.99 0.001"


def _add_joint_damping(root: ET.Element) -> None:
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        if name in ARM_JOINTS:
            joint.attrib["damping"] = "15"
        elif name in ("Claw_left", "claw_right"):
            joint.attrib["damping"] = "30"


def _add_contact_excludes(root: ET.Element) -> None:
    """Disable collisions between claws and hand, and between the two claws."""
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for b1, b2 in [("Claw_Link_left", "Hand_Link"),
                   ("Claw_Link_right", "Hand_Link"),
                   ("Claw_Link_left", "Claw_Link_right")]:
        if contact.find(f".//exclude[@body1='{b1}'][@body2='{b2}']") is None:
            ET.SubElement(contact, "exclude", body1=b1, body2=b2)


def _extend_claw_joint_ranges(root: ET.Element) -> None:
    """Claw joint ranges already correct from URDF [0, 0.04] / [-0.04, 0]."""
    pass


def _add_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")

    for joint in ARM_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            name=f"{joint}_pos",
            joint=joint,
            kp="80",
            ctrlrange="-3.1416 3.1416",
            ctrllimited="true",
        )

    ET.SubElement(
        actuator,
        "position",
        name="Claw_left_pos",
        joint=GRIPPER_JOINT,
        kp="600",
        ctrlrange="0.0 0.04",
        forcerange="-200 200",
        ctrllimited="true",
        forcelimited="true",
    )
    ET.SubElement(
        actuator,
        "position",
        name="Claw_right_pos",
        joint="claw_right",
        kp="600",
        ctrlrange="-0.04 0.0",
        forcerange="-200 200",
        ctrllimited="true",
        forcelimited="true",
    )


def _configure_visual(root: ET.Element, width: int, height: int) -> None:
    visual = _ensure_child(root, "visual")
    global_vis = _ensure_child(visual, "global")
    global_vis.attrib["offwidth"] = str(width)
    global_vis.attrib["offheight"] = str(height)


def build_env(urdf_path: Path, anomaly_stl: Path, normal_stl: Path, out_xml: Path, width: int, height: int) -> None:
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not anomaly_stl.exists():
        raise FileNotFoundError(f"Anomaly STL not found: {anomaly_stl}")
    if not normal_stl.exists():
        raise FileNotFoundError(f"Normal STL not found: {normal_stl}")

    meshes_dir = urdf_path.parent.parent / "meshes"
    root = _compile_urdf_to_mjcf(urdf_path, meshes_dir)

    compiler = _ensure_child(root, "compiler")
    compiler.attrib["angle"] = "radian"

    option = _ensure_child(root, "option")
    option.attrib["timestep"] = "0.002"
    option.attrib["gravity"] = "0 0 -9.81"
    option.attrib["cone"] = "elliptic"
    option.attrib["impratio"] = "10"
    option.attrib["solver"] = "Newton"
    option.attrib["iterations"] = "200"
    option.attrib["noslip_iterations"] = "2"

    asset = _ensure_child(root, "asset")
    worldbody = _ensure_child(root, "worldbody")
    _set_robot_mesh_abs_paths(asset, meshes_dir)
    _add_assets(asset, anomaly_stl, normal_stl)

    _add_cameras_and_sites(worldbody)
    _add_table_and_objects(worldbody)
    _add_friction_to_claw_geoms(worldbody)

    _add_joint_damping(root)
    _add_contact_excludes(root)
    _extend_claw_joint_ranges(root)
    _add_actuators(root)
    _configure_visual(root, width=width, height=height)

    _indent(root)
    tree = ET.ElementTree(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MuJoCo env.xml from Arm_6Dof URDF.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("Arm_6Dof_claw_urdf/urdf/Arm_6Dof.urdf"),
        help="Path to source URDF.",
    )
    parser.add_argument(
        "--anomaly",
        type=Path,
        default=Path("Sample/Anomaly.STL"),
        help="Path to anomaly STL.",
    )
    parser.add_argument(
        "--normal",
        type=Path,
        default=Path("Sample/Normal.STL"),
        help="Path to normal STL.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("env.xml"),
        help="Output MuJoCo XML path.",
    )
    parser.add_argument("--width", type=int, default=256, help="Offscreen render width.")
    parser.add_argument("--height", type=int, default=256, help="Offscreen render height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_env(
        urdf_path=args.urdf,
        anomaly_stl=args.anomaly,
        normal_stl=args.normal,
        out_xml=args.out,
        width=args.width,
        height=args.height,
    )
    print(f"[build_env] wrote {args.out}")


if __name__ == "__main__":
    main()

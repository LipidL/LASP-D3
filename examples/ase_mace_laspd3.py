import numpy as np
import ase
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.langevin import Langevin

from d3_cffi import D3Calculator

# eV/Å³ to GPa
EV_A3_TO_GPA = 160.21766208
PATH_TO_MODEL = './models/mace-medium.model'
PATH_TO_STRUCTURE = './structures/192.extxyz'


class MACED3Calculator(Calculator):
    """ASE Calculator combining MACE with DFT-D3 dispersion correction."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_paths: str,
        device: str = "cuda",
        cutoff_radius: float = 46.4758,
        cn_cutoff_radius: float = 46.4758,
        damping_type: int = 0,
        functional_type: int = 0,
        max_length: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mace_calc = MACECalculator(model_paths=model_paths, device=device)
        self.cutoff_radius = cutoff_radius
        self.cn_cutoff_radius = cn_cutoff_radius
        self.damping_type = damping_type
        self.functional_type = functional_type
        self.max_length = max_length
        self._d3_calc = None

    def _get_d3(self, atoms: ase.Atoms) -> D3Calculator:
        elements = atoms.get_atomic_numbers().tolist()
        n = len(elements)
        max_len = max(self.max_length, n)
        if self._d3_calc is None:
            self._d3_calc = D3Calculator(
                elements=elements,
                max_length=max_len,
                cutoff_radius=self.cutoff_radius,
                cn_cutoff_radius=self.cn_cutoff_radius,
                damping_type=self.damping_type,
                functional_type=self.functional_type,
            )
        return self._d3_calc

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        # --- MACE ---
        self.mace_calc.calculate(atoms=self.atoms, properties=properties)
        mace_energy = self.mace_calc.results["energy"]
        mace_forces = self.mace_calc.results["forces"]
        mace_stress = self.mace_calc.results.get("stress", np.zeros(6))

        # --- D3 ---
        assert isinstance(self.atoms, ase.Atoms)
        d3 = self._get_d3(self.atoms)
        elements = self.atoms.get_atomic_numbers().tolist()
        positions = np.ascontiguousarray(self.atoms.get_positions(), dtype=np.float32)
        cell = np.ascontiguousarray(self.atoms.get_cell().array, dtype=np.float32)

        d3.set_atoms(positions, elements)
        d3.set_cell(cell)
        d3_energy, d3_forces, d3_stress_3x3 = d3.compute()

        # D3 stress (3x3) -> Voigt 6-component, and convert to eV/Å³
        volume = self.atoms.get_volume()
        # d3_stress_3x3 assumed in eV units (energy); convert to stress = val / volume
        d3_stress_voigt = np.array([
            d3_stress_3x3[0, 0],
            d3_stress_3x3[1, 1],
            d3_stress_3x3[2, 2],
            d3_stress_3x3[1, 2],
            d3_stress_3x3[0, 2],
            d3_stress_3x3[0, 1],
        ]) / volume

        # --- Combine ---
        self.results["energy"] = mace_energy + d3_energy
        self.results["forces"] = mace_forces + d3_forces
        self.results["stress"] = mace_stress + d3_stress_voigt


if __name__ == "__main__":
    init_conf = read(PATH_TO_STRUCTURE, "0")
    assert isinstance(init_conf, ase.Atoms)

    calculator = MACED3Calculator(
        model_paths=PATH_TO_MODEL,
        device="cuda",
        max_length=len(init_conf),
    )
    init_conf.calc = calculator

    dyn = Langevin(init_conf, 0.5 * units.fs, temperature_K=310, friction=5e-3)

    def write_frame():
        dyn.atoms.write("result.xyz", append=True)

    dyn.attach(write_frame, interval=50)
    dyn.run(100)
    print("MACE+D3 MD finished!")


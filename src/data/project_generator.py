"""
project_generator.py
--------------------
Generates realistic construction project schedules with:
  - Task list with 3-point duration estimates (optimistic / most-likely / pessimistic)
  - Dependency graph (predecessor relationships)
  - Cost estimates per task
  - Trade / phase classification (TCE: Gros Oeuvre, Second Oeuvre, VRD, etc.)

Project types:
  - residential:      18-task residential building (R+3)
  - commercial:       28-task office/commercial complex
  - infrastructure:   22-task civil/infrastructure project

Output: dict serializable as JSON, consumable by CPM and Monte Carlo engines.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    name: str
    phase: str
    trade: str
    # 3-point duration estimates (working days)
    duration_optimistic: float
    duration_most_likely: float
    duration_pessimistic: float
    # Cost estimates (EUR)
    cost_optimistic: float
    cost_most_likely: float
    cost_pessimistic: float
    # Dependencies (list of task IDs that must finish before this starts)
    predecessors: list[str] = field(default_factory=list)
    # Risk level (for display)
    risk_level: str = "Medium"   # Low / Medium / High

    @property
    def duration_pert(self) -> float:
        return (self.duration_optimistic + 4*self.duration_most_likely + self.duration_pessimistic) / 6

    @property
    def cost_pert(self) -> float:
        return (self.cost_optimistic + 4*self.cost_most_likely + self.cost_pessimistic) / 6


@dataclass
class Project:
    name: str
    project_type: str
    description: str
    tasks: list[Task]
    deadline_days: int | None = None   # contractual deadline
    budget: float | None = None        # contractual budget

    def __post_init__(self) -> None:
        # By default the contract commits to the deterministic plan built from
        # most-likely estimates, as a planner without risk analysis would.
        if self.deadline_days is None:
            self.deadline_days = round(self.planned_duration)
        if self.budget is None:
            self.budget = round(self.planned_cost, -3)

    def _cpm_duration(self, durations: dict[str, float]) -> float:
        from src.schedule.cpm import CPMScheduler

        tasks = [asdict(t) for t in self.tasks]
        return CPMScheduler(tasks, durations=durations).compute().project_duration

    @property
    def planned_duration(self) -> float:
        """Critical-path length with most-likely durations (deterministic plan)."""
        return self._cpm_duration({t.id: t.duration_most_likely for t in self.tasks})

    @property
    def planned_cost(self) -> float:
        return sum(t.cost_most_likely for t in self.tasks)

    @property
    def baseline_duration(self) -> float:
        """Critical-path length with PERT mean durations."""
        return self._cpm_duration({t.id: t.duration_pert for t in self.tasks})

    @property
    def baseline_cost(self) -> float:
        return sum(t.cost_pert for t in self.tasks)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add computed fields
        d["baseline_duration_pert"] = self.baseline_duration
        d["baseline_cost_pert"] = self.baseline_cost
        return d


# ─── Project templates ────────────────────────────────────────────────────────

def build_residential() -> Project:
    """
    R+3 residential building — typical Moroccan/French urban construction.
    18 tasks across 5 phases: VRD, Gros Oeuvre, Second Oeuvre, Finitions, Réception.
    """
    tasks = [
        Task("T01","Installation de chantier","Préparation","Chantier",
             3,5,8,4000,6000,9000,[]),
        Task("T02","Terrassement & VRD","VRD","Terrassement",
             8,12,20,18000,25000,38000,["T01"],"High"),
        Task("T03","Fondations","Gros Oeuvre","Maçonnerie",
             10,15,25,35000,48000,70000,["T02"],"High"),
        Task("T04","Voile RDC","Gros Oeuvre","Béton armé",
             7,10,16,22000,30000,44000,["T03"]),
        Task("T05","Dalle RDC","Gros Oeuvre","Béton armé",
             5,8,12,15000,20000,30000,["T04"]),
        Task("T06","Voile R+1","Gros Oeuvre","Béton armé",
             7,10,16,22000,30000,44000,["T05"]),
        Task("T07","Dalle R+1","Gros Oeuvre","Béton armé",
             5,8,12,15000,20000,30000,["T06"]),
        Task("T08","Voile R+2 & Toiture","Gros Oeuvre","Béton armé",
             10,14,22,32000,42000,62000,["T07"]),
        Task("T09","Maçonnerie cloisons","Second Oeuvre","Maçonnerie",
             12,18,28,20000,28000,42000,["T05"],"High"),
        Task("T10","Plomberie sanitaire","Second Oeuvre","Plomberie",
             10,15,22,18000,24000,36000,["T09"]),
        Task("T11","Électricité CFO/CFA","Second Oeuvre","Électricité",
             12,17,26,22000,30000,45000,["T09"]),
        Task("T12","HVAC","Second Oeuvre","CVC",
             8,12,18,15000,20000,30000,["T09"]),
        Task("T13","Enduits & carrelage","Finitions","Revêtements",
             15,22,35,25000,35000,52000,["T10","T11","T12"],"High"),
        Task("T14","Menuiseries extérieures","Finitions","Menuiserie",
             7,10,16,18000,25000,38000,["T08"]),
        Task("T15","Menuiseries intérieures","Finitions","Menuiserie",
             8,12,18,12000,17000,26000,["T13"]),
        Task("T16","Peinture","Finitions","Peinture",
             10,14,22,14000,19000,29000,["T15"]),
        Task("T17","Aménagements extérieurs","VRD","Espaces verts",
             5,8,14,8000,12000,18000,["T14"]),
        Task("T18","Réception & levée réserves","Réception","Direction",
             3,5,10,3000,5000,8000,["T16","T17"]),
    ]
    p = Project(
        name="Immeuble Résidentiel R+3",
        project_type="residential",
        description="Immeuble résidentiel de 3 étages, 12 appartements, Casablanca",
        tasks=tasks,
    )
    return p


def build_commercial() -> Project:
    """28-task commercial office complex."""
    tasks = [
        Task("T01","Études & permis","Préparation","Bureau d'études",10,15,25,20000,30000,45000,[]),
        Task("T02","Installation de chantier","Préparation","Chantier",3,5,8,5000,7000,11000,["T01"]),
        Task("T03","Démolition existant","Préparation","Démolition",5,8,15,12000,18000,30000,["T02"],"High"),
        Task("T04","VRD primaires","VRD","Terrassement",10,15,25,30000,42000,65000,["T03"],"High"),
        Task("T05","Fondations spéciales","Gros Oeuvre","Fondations",12,18,30,60000,85000,130000,["T04"],"High"),
        Task("T06","Structure métallique R0","Gros Oeuvre","Charpente",10,14,22,55000,75000,110000,["T05"]),
        Task("T07","Structure métallique R1","Gros Oeuvre","Charpente",10,14,22,55000,75000,110000,["T06"]),
        Task("T08","Structure métallique R2","Gros Oeuvre","Charpente",10,14,22,55000,75000,110000,["T07"]),
        Task("T09","Couverture & étanchéité","Gros Oeuvre","Couverture",8,12,20,35000,48000,72000,["T08"]),
        Task("T10","Façade rideau","Second Oeuvre","Façadier",15,22,35,80000,110000,165000,["T08"]),
        Task("T11","Cloisons sèches R0","Second Oeuvre","Plaquiste",8,12,18,20000,28000,42000,["T06"]),
        Task("T12","Cloisons sèches R1","Second Oeuvre","Plaquiste",8,12,18,20000,28000,42000,["T07"]),
        Task("T13","Cloisons sèches R2","Second Oeuvre","Plaquiste",8,12,18,20000,28000,42000,["T08"]),
        Task("T14","Plomberie R0","Second Oeuvre","Plomberie",8,12,18,22000,30000,45000,["T11"]),
        Task("T15","Plomberie R1-R2","Second Oeuvre","Plomberie",10,15,22,30000,42000,63000,["T12","T13"]),
        Task("T16","Électricité R0","Second Oeuvre","Électricité",10,15,22,28000,38000,57000,["T11"]),
        Task("T17","Électricité R1-R2","Second Oeuvre","Électricité",12,18,28,38000,52000,78000,["T12","T13"]),
        Task("T18","CVC & désenfumage","Second Oeuvre","CVC",12,18,28,45000,62000,93000,["T11","T12","T13"],"High"),
        Task("T19","Ascenseur","Second Oeuvre","Ascensoriste",5,8,14,28000,38000,57000,["T06","T07","T08"]),
        Task("T20","Faux plafonds","Finitions","Plaquiste",8,12,18,18000,25000,38000,["T14","T15","T16","T17","T18"]),
        Task("T21","Sols techniques","Finitions","Revêtements",6,9,15,22000,30000,45000,["T20"]),
        Task("T22","Peinture intérieure","Finitions","Peinture",8,12,18,15000,20000,30000,["T21"]),
        Task("T23","Menuiseries bois","Finitions","Menuiserie",6,9,14,18000,24000,36000,["T22"]),
        Task("T24","Signalétique & sécurité","Finitions","Sécurité",4,6,10,12000,16000,24000,["T22"]),
        Task("T25","VRD finales","VRD","Terrassement",6,9,15,15000,20000,30000,["T10"]),
        Task("T26","Espaces verts","Extérieurs","Paysagiste",5,8,14,10000,14000,21000,["T25"]),
        Task("T27","Levée de réserves","Réception","Direction",5,8,15,5000,8000,15000,["T23","T24","T26"]),
        Task("T28","Réception travaux","Réception","Maître d'ouvrage",2,3,6,3000,4000,7000,["T27"]),
    ]
    return Project(
        name="Complexe de Bureaux R+2",
        project_type="commercial",
        description="Immeuble de bureaux 3 niveaux, 2400m², parking souterrain",
        tasks=tasks,
    )


def build_infrastructure() -> Project:
    """22-task civil/infrastructure project (road + utilities)."""
    tasks = [
        Task("T01","Études & autorisations","Études","Bureau d'études",15,20,35,25000,35000,55000,[]),
        Task("T02","Signalisation & déviation","Préparation","Signalisation",5,7,12,8000,11000,18000,["T01"]),
        Task("T03","Décapage terre végétale","Terrassement","Terrassier",5,8,15,12000,17000,28000,["T02"]),
        Task("T04","Terrassement en déblai","Terrassement","Terrassier",15,22,38,45000,65000,105000,["T03"],"High"),
        Task("T05","Remblai compacté","Terrassement","Terrassier",10,15,25,30000,42000,68000,["T04"]),
        Task("T06","Drainage & assainissement","VRD","Canalisateur",12,18,30,38000,55000,88000,["T04"],"High"),
        Task("T07","Alimentation eau potable","Réseaux","Canalisateur",8,12,20,22000,30000,48000,["T06"]),
        Task("T08","Réseaux électricité HTA","Réseaux","Électricien",10,15,25,35000,48000,78000,["T05"]),
        Task("T09","Éclairage public","Réseaux","Électricien",8,12,20,20000,28000,45000,["T08"]),
        Task("T10","Télécoms & fibre","Réseaux","Télécom",6,9,15,15000,20000,32000,["T05"]),
        Task("T11","Fondation de chaussée","Chaussée","Enrobés",10,14,22,40000,55000,85000,["T05","T06"],"High"),
        Task("T12","Couche de base","Chaussée","Enrobés",8,12,20,35000,48000,75000,["T11"]),
        Task("T13","Couche de roulement","Chaussée","Enrobés",5,8,14,28000,38000,60000,["T12"],"High"),
        Task("T14","Trottoirs & bordures","Aménagement","Maçonnerie",8,12,20,18000,25000,40000,["T13"]),
        Task("T15","Ouvrages d'art — semelles","OA","Génie civil",12,18,30,55000,78000,125000,["T04"],"High"),
        Task("T16","Ouvrages d'art — piles","OA","Génie civil",15,22,38,75000,105000,170000,["T15"],"High"),
        Task("T17","Ouvrages d'art — tablier","OA","Génie civil",12,18,30,65000,92000,148000,["T16"],"High"),
        Task("T18","Garde-corps & équipements","OA","Métallerie",5,8,14,15000,20000,32000,["T17"]),
        Task("T19","Signalisation horizontale","Finitions","Signalisation",3,5,9,8000,11000,18000,["T13","T14"]),
        Task("T20","Signalisation verticale","Finitions","Signalisation",3,5,9,6000,8000,13000,["T14"]),
        Task("T21","Espaces verts & talus","Finitions","Paysagiste",6,9,16,10000,14000,22000,["T14"]),
        Task("T22","Réception & essais","Réception","Maître d'ouvrage",4,6,12,8000,11000,18000,["T18","T19","T20","T21"]),
    ]
    return Project(
        name="Voirie & Réseaux Divers — Zone Industrielle",
        project_type="infrastructure",
        description="VRD et ouvrages d'art, 3.2km de voirie, zone industrielle",
        tasks=tasks,
    )


# ─── Builder ──────────────────────────────────────────────────────────────────

BUILDERS = {
    "residential":    build_residential,
    "commercial":     build_commercial,
    "infrastructure": build_infrastructure,
}


def generate_project(project_type: str = "residential") -> Project:
    builder = BUILDERS.get(project_type)
    if builder is None:
        raise ValueError(f"Unknown project type: {project_type}. Choose from {list(BUILDERS)}")
    return builder()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate construction project")
    parser.add_argument("--type", choices=list(BUILDERS), default="residential")
    parser.add_argument("--output", default="data/project.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    project = generate_project(args.type)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(project.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"✅ Project '{project.name}' generated: {len(project.tasks)} tasks")
    print(f"   Plan (most likely):       {project.planned_duration:.0f} days, €{project.planned_cost:,.0f}")
    print(f"   Baseline (PERT means):    {project.baseline_duration:.1f} days, €{project.baseline_cost:,.0f}")
    print(f"   Saved to: {args.output}")

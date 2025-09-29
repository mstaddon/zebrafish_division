# Unilateral cell division in the zebrafish egg

## division_simulation.py

This code simulates cell division in the zebrafish where only a partial contractile cable forms around the cell. The cell is modeled as a spherical mesh of edges that behave like a Burger's material, meaning friction at short time-scales, elastic at intermediate time-scales, and viscous at long time-scales. The band grows around the equator of the cell but also generates contractile forces. In this simulation, the cell also undergoes two phases with different material properties.

## PIV_analysis.py

This code analysis PIV data from dividing cells to estimate the growth rate of the band and the flow speed of the cytoskeleton. It roughly tracks the end points of the band by assuming that the flow is highest at the end of the band.

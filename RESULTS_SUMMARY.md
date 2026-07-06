# Results Summary

## Part 1 - Oriented Bounding Boxes

| Mesh | Length | Width | Height | OBB volume |
|---|---:|---:|---:|---:|
| CUBE.obj | 13.055800 | 13.055800 | 12.525200 | 2134.969531 |
| CYLINDER.obj | 3.999100 | 3.896809 | 3.896570 | 60.723094 |
| TEAPOT.obj | 2.408144 | 1.981797 | 1.643195 | 7.842072 |

Dimensions are in the same units as the OBJ coordinates.

## Part 2 - Packing

- Master box: `100 x 100 x 100`
- Items placed: `20/20`
- Occupied bounds: `100 x 80 x 30`
- Maximum occupied height: `30`
- Height lower bound: `30`
- Height-optimal: `Yes`
- No overlaps: `Validated`
- Gravity/full support: `Validated`
- Total item volume: `168250`
- Master-box utilization: `16.825%`
- Occupied-envelope utilization: `70.104%`

Exact coordinates and rotations are in `outputs/packing_solution.csv` and `outputs/packing_solution.json`.

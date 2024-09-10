from hotspots_simulation.render_hotspots_patrols_routing import RenderHotpotsPatrolRouting

from hotspots_simulation.environment.environment import EnvironmentStreets


if __name__ == '__main__':
    zona = 3
    w = EnvironmentStreets(zona, 50, clean_not_accessible=True)
    r = RenderHotpotsPatrolRouting(w, render_size=(1020, 1020))
    while True:
        r.render('human_off')

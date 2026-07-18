"""One-off script to capture live-execution screenshots of HARP-WDN for the
Project-I report. Drives the already-running frontend (localhost:5173)
against the already-running backend (localhost:8000)."""
import time
from playwright.sync_api import sync_playwright

OUT = "/Users/anjalisehrawat/Desktop/WDS/docs/screenshots"
BASE = "http://localhost:5173"


def shot(page, name):
    page.screenshot(path=f"{OUT}/{name}.png")
    print("saved", name)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000})

    # 1. Empty state
    page.goto(BASE, wait_until="networkidle")
    time.sleep(1)
    shot(page, "01_simulation_empty")

    # 2. Select Bengaluru
    page.select_option("select.form-control", "bangalore")
    time.sleep(2.5)
    shot(page, "02_simulation_bangalore_loaded")

    # 3. Select Pipe Burst scenario
    page.get_by_text("Pipe Burst", exact=True).click()
    time.sleep(0.5)

    # 4. Pick a failed pipe from the pipe select (second <select>)
    selects = page.query_selector_all("select.form-control")
    pipe_select = None
    for s in selects:
        opts = s.query_selector_all("option")
        texts = [o.inner_text() for o in opts]
        if any("→" in t for t in texts):  # "P3 (X → Y)"
            pipe_select = s
            break
    if pipe_select:
        opt_values = [o.get_attribute("value") for o in pipe_select.query_selector_all("option") if o.get_attribute("value")]
        if opt_values:
            pipe_select.select_option(opt_values[0])
    time.sleep(0.5)
    shot(page, "03_pipe_burst_selected")

    # 5. Compare Both
    page.get_by_text("Compare Both", exact=True).click()
    time.sleep(3)
    shot(page, "04_compare_running_map")

    # 6. Stats tab (flow animation / metrics panel)
    try:
        page.get_by_text("Stats", exact=False).first.click()
        time.sleep(1)
        shot(page, "05_stats_tab")
    except Exception as e:
        print("stats tab click failed:", e)

    # 7. Results tab (client-side nav — keeps lifted sharedSim state).
    # Scoped to .nav-links (NOT nav.top-nav) because the brand subtitle text
    # "WATER NETWORK SYSTEM" also contains "Network" and would match first.
    page.click(".nav-links >> text=Results")
    time.sleep(1.5)
    shot(page, "06_results_overview")

    # 8. Results -> try clicking sub-tabs if present (Nodes, Charts, Compare)
    for label in ["Nodes", "Charts", "Compare"]:
        try:
            el = page.locator(f"button:has-text('{label}'), .tab:has-text('{label}')").first
            el.click(timeout=5000)
            time.sleep(1)
            shot(page, f"06_results_{label.lower()}")
        except Exception as e:
            print(f"results sub-tab {label} failed:", e)

    # 9. Network tab
    page.click(".nav-links >> text=Network")
    time.sleep(1.5)
    shot(page, "07_network_view")

    # 10. Dark mode toggle (stay on Network tab, still has data loaded)
    time.sleep(0.5)
    try:
        page.locator(".theme-toggle").click()
        time.sleep(0.5)
        shot(page, "08_dark_mode")
    except Exception as e:
        print("theme toggle failed:", e)

    browser.close()

print("DONE")

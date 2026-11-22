"""Extract standalone vector figures from the saved main benchmark PDF.

Run with the system Python (PyGObject, Poppler and Cairo are required).
No experiment data, policy evaluation or simulation is used.
"""

from pathlib import Path

import cairo
import gi

gi.require_version("Poppler", "0.18")
from gi.repository import Poppler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/figures/main_benchmarks.pdf"
OUTPUT = SOURCE.parent


def main():
    document = Poppler.Document.new_from_file(SOURCE.as_uri(), None)
    page = document.get_page(0)

    for name, left, right, legend_names in [
        ("distribution", 114, 212, ["reference", "transport", "mf", "reinforce"]),
        ("portfolio", 315, 416, ["reference", "transport", "proba", "reinforce"]),
        ("bimodal", 416, 515, ["reference", "transport", "single", "reinforce", "normal"]),
    ]:
        for panel in ("validation", "population"):
            suffix = "validation_gap" if panel == "validation" else ("mean_wealth_flow" if name == "portfolio" else "terminal_law")
            destination = OUTPUT / f"main_benchmark_{name}_{suffix}.pdf"
            surface = cairo.PDFSurface(str(destination), 250, 300)
            surface.set_metadata(cairo.PDF_METADATA_TITLE, f"{name.title()}: {suffix.replace(chr(95), chr(32))}")
            surface.set_metadata(cairo.PDF_METADATA_SUBJECT, f"Extracted from {SOURCE.name}; no recomputation")
            context = cairo.Context(surface)
            context.set_source_rgb(1, 1, 1)
            context.paint()

            def crop(box, x, y, scale=2):
                x0, y0, x1, y1 = box
                context.save()
                context.translate(x, y)
                context.scale(scale, scale)
                context.rectangle(0, 0, x1 - x0, y1 - y0)
                context.clip()
                context.translate(-x0, -y0)
                page.render_for_printing(context)
                context.restore()

            def text(label, x, y, size=8):
                context.set_source_rgb(0.2, 0.2, 0.2)
                context.select_font_face("Sans")
                context.set_font_size(size)
                context.move_to(x, y)
                context.show_text(label)

            panel_legend = list(legend_names)
            if panel == "validation":
                # The shared axis labels occur only once in the source figure.
                crop((left, 0, right, 95), 37, 8)
                crop((18, 28, 33, 70), 6, 63)
                crop((231, 95, 331, 106), 37, 201)
                text("Five paired seeds (mean ± std.)", 56, 235, 8)
                panel_legend = [label for label in legend_names if label not in {"reference", "normal"}]
            else:
                crop((left, 106, right, 210), 37, 8)
                text(f"{name.title()}: final policy, seed 0", 53, 235, 8)

            legend_boxes = {
                "reference": (7, 214, 40, 234),
                "transport": (51, 214, 111, 234),
                "single": (120, 214, 200, 234),
                "proba": (210, 214, 294, 234),
                "mf": (303, 214, 389, 234),
                "reinforce": (398, 214, 469, 234),
                "normal": (478, 214, 524, 234),
            }
            for index, label in enumerate(panel_legend):
                crop(legend_boxes[label], 14 + (index % 2) * 119,
                     249 + (index // 2) * 15, scale=1.15)
            context.show_page()
            surface.finish()
            print(destination)

        destination = OUTPUT / f"main_benchmark_{name}_side_by_side.pdf"
        surface = cairo.PDFSurface(str(destination), 500, 285)
        surface.set_metadata(cairo.PDF_METADATA_TITLE, f"{name.title()}: validation and population")
        surface.set_metadata(cairo.PDF_METADATA_SUBJECT, f"Extracted from {SOURCE.name}; no recomputation")
        context = cairo.Context(surface)
        context.set_source_rgb(1, 1, 1)
        context.paint()
        population = "mean_wealth_flow" if name == "portfolio" else "terminal_law"
        for index, suffix in enumerate(("validation_gap", population)):
            source = OUTPUT / f"main_benchmark_{name}_{suffix}.pdf"
            panel_document = Poppler.Document.new_from_file(source.as_uri(), None)
            context.save()
            context.translate(index * 250, 0)
            # Keep the panel and seed annotation; omit its individual legend.
            context.rectangle(0, 0, 250, 242)
            context.clip()
            panel_document.get_page(0).render_for_printing(context)
            context.restore()
        legend_scale = 1.25
        legend_spacing = 16
        legend_width = sum(
            (legend_boxes[label][2] - legend_boxes[label][0]) * legend_scale
            for label in legend_names
        ) + legend_spacing * (len(legend_names) - 1)
        legend_x = (500 - legend_width) / 2
        for label in legend_names:
            box = legend_boxes[label]
            crop(box, legend_x, 250, scale=legend_scale)
            legend_x += (box[2] - box[0]) * legend_scale + legend_spacing
        context.show_page()
        surface.finish()
        print(destination)



if __name__ == "__main__":
    main()

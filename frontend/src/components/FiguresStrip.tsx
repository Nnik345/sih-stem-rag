import { useEffect, useState } from "react";
import type { RunTrace } from "../types";

interface Figure {
  image_id: string;
  page_number?: number;
  score?: number;
  source?: string;
}

function figureSrc(imageId: string): string {
  return `/api/curriculum-images/${encodeURIComponent(imageId)}`;
}

export function FiguresStrip({
  figures,
  title = "Textbook figures",
}: {
  figures: Figure[];
  title?: string;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const openFigure = figures.find((figure) => figure.image_id === openId) ?? null;

  useEffect(() => {
    if (!openId) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpenId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openId]);

  if (!figures.length) return null;
  return (
    <section className="figures-strip" aria-label={title}>
      <h3>{title}</h3>
      <p className="muted">Official NCERT extracts only. Student photos are never stored as curriculum figures. Click a figure to enlarge it.</p>
      <div className="figure-row">
        {figures.map((figure) => (
          <figure key={figure.image_id} className="figure-thumb">
            <button
              type="button"
              className="figure-open"
              onClick={() => setOpenId(figure.image_id)}
              aria-label={`Enlarge textbook figure ${figure.page_number != null ? `page ${figure.page_number}` : figure.image_id}`}
            >
              <img
                src={figureSrc(figure.image_id)}
                alt={`Textbook figure ${figure.image_id}`}
              />
            </button>
            <figcaption>
              {figure.page_number != null ? `p.${figure.page_number}` : figure.image_id}
            </figcaption>
          </figure>
        ))}
      </div>
      {openFigure ? (
        <div
          className="figure-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label="Enlarged textbook figure"
          onClick={() => setOpenId(null)}
        >
          <img
            src={figureSrc(openFigure.image_id)}
            alt={`Textbook figure ${openFigure.image_id}`}
            onClick={(event) => event.stopPropagation()}
          />
          <button type="button" className="figure-lightbox-close" onClick={() => setOpenId(null)}>
            Close
          </button>
        </div>
      ) : null}
    </section>
  );
}

export function figuresFromTrace(trace: RunTrace | null): Figure[] {
  if (!trace) return [];
  if (trace.attached_figures?.length) return trace.attached_figures;
  return trace.evidence && "attached_figures" in trace.evidence
    ? ((trace.evidence as { attached_figures?: Figure[] }).attached_figures ?? [])
    : [];
}

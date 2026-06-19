/* Minimal Plotly type declarations for plotly.js-dist-min */
declare module "plotly.js-dist-min" {
  namespace Plotly {
    interface PlotData {
      x?: unknown[];
      y?: unknown[];
      z?: unknown[][];
      name?: string;
      type?: string;
      mode?: string;
      text?: string | string[] | string[][];
      hoverinfo?: string;
      hovertemplate?: string;
      marker?: Record<string, unknown>;
      line?: Record<string, unknown>;
      fillcolor?: string;
      box?: Record<string, unknown>;
      meanline?: Record<string, unknown>;
      boxpoints?: string;
      colorscale?: [number, string][];
      colorbar?: Record<string, unknown>;
      zmin?: number;
      zmax?: number;
      hoverongaps?: boolean;
      textposition?: string;
      textfont?: Record<string, unknown>;
      orientation?: string;
      node?: Record<string, unknown>;
      link?: Record<string, unknown>;
      [key: string]: unknown;
    }

    interface Layout {
      title?: string | Record<string, unknown>;
      xaxis?: Record<string, unknown>;
      yaxis?: Record<string, unknown>;
      paper_bgcolor?: string;
      plot_bgcolor?: string;
      font?: Record<string, unknown>;
      height?: number;
      width?: number;
      margin?: Record<string, unknown>;
      shapes?: Shape[];
      annotations?: Record<string, unknown>[];
      barmode?: string;
      showlegend?: boolean;
      legend?: Record<string, unknown>;
      [key: string]: unknown;
    }

    interface Shape {
      type?: string;
      x0?: number;
      x1?: number;
      y0?: number;
      y1?: number;
      line?: Record<string, unknown>;
      xref?: string;
      yref?: string;
      [key: string]: unknown;
    }

    interface Config {
      responsive?: boolean;
      displayModeBar?: boolean;
      [key: string]: unknown;
    }

    interface PlotlyHTMLElement extends HTMLDivElement {
      on(event: string, callback: (data: PlotMouseEvent) => void): void;
    }

    interface PlotMouseEvent {
      points: PlotDatum[];
    }

    interface PlotDatum {
      pointIndex: number | [number, number];
      x?: unknown;
      y?: unknown;
      z?: unknown;
      text?: string;
      data?: PlotData;
      [key: string]: unknown;
    }

    type ColorScale = [number, string][];
    type PlotType = string;
    type Annotations = Record<string, unknown>;

    function newPlot(
      root: string | HTMLDivElement,
      data: Partial<PlotData>[],
      layout?: Partial<Layout>,
      config?: Partial<Config>,
    ): Promise<PlotlyHTMLElement>;
  }

  export default Plotly;
}

import type * as React from "react";
type MathProps = React.HTMLAttributes<MathMLElement>;
declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      math: MathProps & { display?: "block" | "inline" };
      mi: MathProps;
      mo: MathProps;
      mn: MathProps;
      mrow: MathProps;
      msub: MathProps;
      msup: MathProps;
      mover: MathProps;
      mfrac: MathProps;
      munder: MathProps;
      munderover: MathProps;
      mtext: MathProps;
    }
  }
}
export {};

"""A 27-second walkthrough of three-tier routing: Jev reads each request, then Sonnet 5, Opus 5 or a human takes it.

The numbers on screen come from the recorded run in routing.ipynb (40 requests, 40 Jev calls).

    manim -qh --fps 60 animation/routing_flow.py RoutingFlow
"""

from manim import (
    DOWN, LEFT, RIGHT, UP, AnimationGroup, Circle, Create, Dot, FadeIn, FadeOut, GrowFromCenter, Indicate,
    LaggedStart, Line, MoveAlongPath, Rectangle, RoundedRectangle, Scene, ShowPassingFlash, Succession, Text,
    VGroup, Wait, config, rate_functions,
)
from manim.utils.color import ManimColor

config.background_color = ManimColor("#0B0F19")

FONT = "Avenir Next"
INK, MUTED, FAINT = ManimColor("#E8ECF4"), ManimColor("#8A93A6"), ManimColor("#2A3244")
JEV, SONNET, OPUS, HUMAN = ManimColor("#18C29C"), ManimColor("#4F8CFF"), ManimColor("#A06BFF"), ManimColor("#FFB547")

# From the recorded run: Jev's probabilities, confidence and one-way-door flag for three of the 40 requests.
EXAMPLES = [
    ("How do I rotate an IAM access key\nwithout breaking CI?", {"general": 0.87, "big decision": 0.13, "one-way door": 0.00},
     0.07, "sonnet", "general question, 80% sure", "Sonnet 5 answers  ·  5.0 s  ·  $0.003"),
    ("Error rate 2.1% vs 0.3% baseline.\nRoll back or hotfix forward?", {"general": 0.00, "big decision": 1.00, "one-way door": 0.00},
     0.48, "opus", "big decision, reversible", "Opus 5 answers  ·  5.5 s  ·  $0.008"),
    ("Pick a vendor for a 3-year,\n$2M observability contract", {"general": 0.00, "big decision": 0.02, "one-way door": 0.98},
     0.94, "human", "one-way door: contract", "Claude call cancelled  ·  ticket filed"),
]
ROUTED = {"sonnet": 20, "opus": 11, "human": 9}  # where the 40 requests went


def label(text: str, size: int = 24, color=INK, weight: str = "NORMAL") -> Text:
    return Text(text, font=FONT, font_size=size, color=color, weight=weight)


class RoutingFlow(Scene):
    def construct(self) -> None:
        self.title()
        self.stage()
        for example in EXAMPLES:
            self.route_one(*example)
        self.burst()
        self.results()

    # 0 to 3 s
    def title(self) -> None:
        head = label("Jev routes. Claude answers. Humans decide.", 44, weight="BOLD")
        sub = label("Three-tier LLM routing with Strands Agents hooks on Amazon Bedrock", 24, MUTED)
        group = VGroup(head, sub).arrange(DOWN, buff=0.35)
        self.play(FadeIn(head, shift=UP * 0.3), run_time=0.9)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.wait(0.8)
        self.play(group.animate.scale(0.55).to_edge(UP, buff=0.3), run_time=0.8)
        self.header = group

    # 3 to 6 s
    def stage(self) -> None:
        self.jev_ring = Circle(radius=0.95, color=JEV, stroke_width=4).move_to(LEFT * 1.6)
        glow = Circle(radius=1.25, color=JEV, stroke_width=14, stroke_opacity=0.18).move_to(self.jev_ring)
        jev_text = VGroup(label("Jev", 34, JEV, "BOLD"), label("one typed call", 16, MUTED)).arrange(DOWN, buff=0.08)
        jev_text.move_to(self.jev_ring)
        self.jev = VGroup(glow, self.jev_ring, jev_text)

        self.lanes = {}
        specs = [("sonnet", "Claude Sonnet 5", "general questions", SONNET, 2.1),
                 ("opus", "Claude Opus 5", "big, reversible decisions", OPUS, 0.0),
                 ("human", "A human", "one-way doors", HUMAN, -2.1)]
        for key, name, what, color, y in specs:
            box = RoundedRectangle(corner_radius=0.18, width=4.3, height=1.35, stroke_color=color, stroke_width=3,
                                   fill_color=color, fill_opacity=0.08).move_to(RIGHT * 4.3 + UP * y)
            text = VGroup(label(name, 26, color, "BOLD"), label(what, 17, MUTED)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            text.move_to(box).align_to(box, LEFT).shift(RIGHT * 0.3)
            path = Line(self.jev_ring.get_right(), box.get_left(), stroke_color=FAINT, stroke_width=3)
            self.lanes[key] = {"box": box, "text": text, "path": path, "color": color}

        intake = label("incoming requests", 18, MUTED).move_to(LEFT * 5.3 + UP * 1.35)
        self.play(GrowFromCenter(self.jev), FadeIn(intake), run_time=0.8)
        self.play(LaggedStart(*[AnimationGroup(Create(l["path"]), FadeIn(l["box"], shift=LEFT * 0.2), FadeIn(l["text"]))
                                for l in self.lanes.values()], lag_ratio=0.25), run_time=1.3)
        self.intake = intake

    # about 4 s per request
    def route_one(self, text, probs, one_way, lane, rule, outcome) -> None:
        card = RoundedRectangle(corner_radius=0.15, width=3.9, height=1.25, stroke_color=INK, stroke_width=1.5,
                                fill_color=ManimColor("#151B2B"), fill_opacity=1).move_to(LEFT * 5.3 + UP * 0.3)
        words = label(text, 18).move_to(card)
        request = VGroup(card, words)
        self.play(FadeIn(request, shift=RIGHT * 0.4), run_time=0.45)
        self.play(request.animate.scale(0.25).move_to(self.jev_ring).set_opacity(0), Indicate(self.jev_ring, color=JEV, scale_factor=1.08),
                  run_time=0.6)
        self.remove(request)

        bars = VGroup()
        for name, p in probs.items():
            track = Rectangle(width=2.2, height=0.16, stroke_width=0, fill_color=FAINT, fill_opacity=1)
            fill = Rectangle(width=max(2.2 * p, 0.02), height=0.16, stroke_width=0, fill_color=JEV, fill_opacity=1)
            fill.align_to(track, LEFT)
            row = VGroup(label(name, 15, MUTED), VGroup(track, fill), label(f"{p:.0%}", 15, INK))
            row.arrange(RIGHT, buff=0.18)
            bars.add(row)
        flag = label(f"one-way-door flag  {one_way:.2f}", 15, HUMAN if one_way >= 0.5 else MUTED)
        readout = VGroup(*bars, flag).arrange(DOWN, buff=0.14, aligned_edge=RIGHT).next_to(self.jev_ring, DOWN, buff=0.45)
        chip = label(rule, 17, self.lanes[lane]["color"], "BOLD").next_to(self.jev_ring, UP, buff=0.35)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in readout], lag_ratio=0.15), run_time=0.7)
        self.play(FadeIn(chip, shift=DOWN * 0.1), run_time=0.35)

        target = self.lanes[lane]
        dot = Dot(self.jev_ring.get_right(), radius=0.11, color=target["color"])
        flash = target["path"].copy().set_stroke(target["color"], width=6)
        self.play(MoveAlongPath(dot, target["path"]), ShowPassingFlash(flash, time_width=0.6), run_time=0.75,
                  rate_func=rate_functions.ease_in_out_sine)
        result = label(outcome, 16, INK).next_to(target["box"], DOWN, buff=0.08).align_to(target["box"], LEFT)
        self.play(FadeOut(dot, scale=2.5), Indicate(target["box"], color=target["color"], scale_factor=1.04), FadeIn(result), run_time=0.55)
        self.wait(0.35)
        self.play(FadeOut(readout), FadeOut(chip), FadeOut(result), run_time=0.35)

    # about 3 s: all 40 requests
    def burst(self) -> None:
        caption = label("all 40 requests  ·  40 Jev calls, cached", 20, MUTED).next_to(self.jev_ring, DOWN, buff=0.5)
        self.play(FadeIn(caption), run_time=0.3)
        order = [k for k, n in ROUTED.items() for _ in range(n)]
        order = [order[i] for i in (list(range(0, 40, 3)) + list(range(1, 40, 3)) + list(range(2, 40, 3)))]
        badges = {k: label("0", 30, self.lanes[k]["color"], "BOLD").move_to(self.lanes[k]["box"]).align_to(self.lanes[k]["box"], RIGHT).shift(LEFT * 0.3)
                  for k in ROUTED}
        self.play(*[FadeIn(b) for b in badges.values()], run_time=0.25)
        flights = []
        for k in order:
            dot = Dot(self.jev_ring.get_right(), radius=0.06, color=self.lanes[k]["color"])
            flights.append(Succession(MoveAlongPath(dot, self.lanes[k]["path"], run_time=0.5), FadeOut(dot, run_time=0.05)))
        self.play(LaggedStart(*flights, lag_ratio=0.05), run_time=2.4)
        for k, n in ROUTED.items():
            badges[k].become(label(str(n), 30, self.lanes[k]["color"], "BOLD").move_to(badges[k]))
        self.play(*[Indicate(b, color=self.lanes[k]["color"]) for k, b in badges.items()], run_time=0.5)
        self.play(FadeOut(caption), run_time=0.2)
        self.badges = badges

    # about 5 s
    def results(self) -> None:
        scene = VGroup(self.jev, self.intake, *[VGroup(l["box"], l["text"], l["path"]) for l in self.lanes.values()], *self.badges.values())
        self.play(FadeOut(scene), run_time=0.6)
        stats = VGroup()
        for big, small, color in [("58%", "cheaper than\nall Opus", JEV), ("0 of 8", "one-way doors\nreached a model", HUMAN),
                                  ("98%", "of requests\nrouted correctly", SONNET)]:
            stats.add(VGroup(label(big, 60, color, "BOLD"), label(small, 22, INK)).arrange(DOWN, buff=0.2))
        stats.arrange(RIGHT, buff=1.4).shift(UP * 0.25)
        footer = VGroup(
            label("40 requests  ·  Claude Sonnet 5 and Opus 5 on Amazon Bedrock  ·  Jev: $0.0008 in total", 18, MUTED),
            label("github.com/mani-aiml/bedrock-strands-decision-models", 18, JEV),
        ).arrange(DOWN, buff=0.18).next_to(stats, DOWN, buff=0.8)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.3) for s in stats], lag_ratio=0.25), run_time=1.4)
        self.play(FadeIn(footer), run_time=0.5)
        self.wait(2.2)

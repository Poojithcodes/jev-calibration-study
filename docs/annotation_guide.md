# AG-Fresh-2026 annotation guide

Label each item with the **main topic** of the news story, using AG News's four classes.
Work alone. Do not look up the article, and do not discuss items until both sheets are finished:
the paper reports agreement between two independent annotators.

| Label | Use it for |
| --- | --- |
| World | International or national politics, government, elections, wars and conflicts, diplomacy, crime and disasters as public events |
| Sports | Matches, results, players, teams, transfers, coaches, sports governing bodies |
| Business | Companies, earnings, markets, the economy, jobs, prices, trade, deals, regulation of business |
| Sci/Tech | Science research, space, technology products, software, internet, AI, gadgets, health research and medicine |
| none | Fits none of the four: entertainment, celebrity, lifestyle, food, travel, opinion pieces with no clear topic, quizzes, live blogs with no single story |

## Tie-breaking rules

1. **Pick what the story is about, not who is in it.** A tech company's quarterly profit is Business. A tech company's new phone is Sci/Tech.
2. **Politics about a domain goes to the domain only when the domain is the point.** A government budget is World. A tax on chip imports is Business.
3. **Sports business** (club takeovers, broadcast deals) is Sports unless it reads as a financial-markets story.
4. **Health:** medical research and new treatments are Sci/Tech. Health policy and outbreaks as public events are World.
5. **Climate and environment:** scientific findings are Sci/Tech. Summits, policy and disasters are World.
6. If two labels are genuinely equal, choose one, put `y` in `unsure`, and add a few words in `note`.

`none` items are removed from the test set after adjudication. They are not errors.

## Process

1. Fill `label` for every row, using exactly one of: `World`, `Sports`, `Business`, `Sci/Tech`, `none`.
2. Save as CSV with the same filename.
3. Run `python scripts/agreement.py` to get Cohen's kappa and the list of disagreements.
4. Resolve every disagreement together in `adjudication.csv` (`final_label` column), then run `python scripts/agreement.py --finalise`.

Aim for about 600 items labelled, so that about 500 survive after `none` items are dropped. At a typical pace that's 2–3 hours each.

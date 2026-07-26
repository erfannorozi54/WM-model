# Speaker Notes: Exact Neural Mass Model for Synaptic-Based Working Memory

Use these notes in presenter mode or as a rehearsal script. They are intentionally concise: each slide has a short purpose, a suggested spoken explanation, and a transition cue.

## Slide 1: Exact Neural Mass Model for Synaptic-Based Working Memory

Purpose: Introduce the paper and the central contribution.

Speaker note:
This paper asks how working memory can be modeled at the population level without simulating every neuron. The main contribution is an exact neural mass model derived from spiking neurons, with short-term synaptic plasticity included. The key idea is that working memory can be stored in synapses, not only in persistent firing.

Transition:
Start by contrasting this with the classical view of working memory.

## Slide 2: 1 - Background

Purpose: Mark the beginning of the motivation section.

Speaker note:
The first part of the talk explains why the paper was needed. I will move from the classical persistent-spiking theory to the synaptic theory of working memory, then explain the missing modeling piece.

Transition:
First, the classical view.

## Slide 3: Classical View: Persistent Spiking

Purpose: Define the traditional working-memory mechanism.

Speaker note:
The standard view is that a memory is kept alive by continuous activity in a population of neurons. After the stimulus disappears, the relevant population keeps firing during the delay. This view was influential because it matched early recordings from prefrontal cortex and was easy to formalize in recurrent network models.

Transition:
But persistent spiking has several empirical and computational problems.

## Slide 4: Why Persistent Spiking Is Not Enough

Purpose: Explain why the classical view is incomplete.

Speaker note:
The problem is not that persistent activity never happens. The problem is that it does not explain all working-memory observations. Trial averaging can make sparse activity look continuous, only a small number of neurons show true persistent firing, and maintaining memories by continuous firing is metabolically costly. It can also create interference between stored items.

Transition:
This motivates an alternative: memory stored in short-term synaptic changes.

## Slide 5: Synaptic Theory of Working Memory

Purpose: Introduce the synaptic alternative.

Speaker note:
In the synaptic theory, the memory trace is stored in temporary changes in synaptic efficacy. A stimulus causes a burst, the burst changes synaptic variables, and the network can remain mostly quiet afterward. Later, a weak cue or a spontaneous burst can reactivate the facilitated population.

Transition:
The issue is that earlier synaptic models were often heuristic.

## Slide 6: The Gap This Paper Fills

Purpose: State the modeling gap.

Speaker note:
Mongillo and colleagues proposed a powerful synaptic mechanism, but the firing-rate models used for this class of mechanism were approximate. They usually tracked firing rate but not mean membrane potential. That matters because many experimental measurements, such as EEG and LFP, are closer to population voltage than to firing rate.

Transition:
The paper's innovation is an exact population model that keeps these missing variables.

## Slide 7: 2 - Main Innovation

Purpose: Mark the contribution section.

Speaker note:
Now I will describe what is mathematically new in the paper: the exact neural mass reduction, the four population variables, and why mean voltage is important.

Transition:
First, what exactly is new?

## Slide 8: What Is New?

Purpose: Summarize the contribution.

Speaker note:
The paper combines three ingredients: exact neural mass theory, heterogeneous QIF neurons, and short-term synaptic plasticity. The result is a low-dimensional model that still reproduces the population dynamics of a very large spiking network. This is not just faster simulation; it also gives access to mean membrane voltage.

Transition:
The next slide explains what is compressed and what variables remain.

## Slide 9: Compression Without Losing the Dynamics

Purpose: Explain the reduced state variables.

Speaker note:
The microscopic network has one voltage equation per neuron, so direct simulation is huge. The exact reduction replaces that with a few variables per population. The fast variables are firing rate, r_k, and mean voltage, v_k. The slow synaptic variables are resources, x_k, and utilization, u_k. The subscript k simply means the item-specific excitatory population.

Transition:
Mean voltage is especially important because it connects the model to experiments.

## Slide 10: Why Mean Membrane Potential Matters

Purpose: Explain why voltage is a major advantage.

Speaker note:
Firing rate tells us how active the population is, but it is not the only relevant macroscopic signal. Mean membrane potential is closer to field-potential measurements. This lets the model make contact with EEG, LFP, and ERP-like signals, which rate-only heuristic models cannot directly provide.

Transition:
Now we move from the innovation to the underlying neuron and synapse model.

## Slide 11: 3 - Building Blocks

Purpose: Begin the mathematical setup.

Speaker note:
This section explains the microscopic ingredients: the QIF neuron, the heterogeneity distribution, and the short-term synaptic plasticity variables.

Transition:
The neuron model is the quadratic integrate-and-fire neuron.

## Slide 12: Quadratic Integrate-and-Fire Neurons

Purpose: Present the microscopic neuron equation.

Speaker note:
Each neuron follows QIF dynamics. The terms include intrinsic voltage dynamics, individual excitability, background current, stimulus current, and recurrent synaptic input. The key point is that this is a spiking model, not a phenomenological rate equation.

Transition:
The exact reduction depends on the way heterogeneity is represented.

## Slide 13: Heterogeneity Makes the Reduction Exact

Purpose: Explain the Lorentzian distribution.

Speaker note:
Neurons are not identical, so each neuron has a different excitability. The model assumes these excitabilities follow a Lorentzian distribution. This is biologically reasonable as heterogeneity, and mathematically crucial because it enables the exact Ott-Antonsen reduction.

Transition:
The second building block is short-term synaptic plasticity.

## Slide 14: Short-Term Synaptic Plasticity

Purpose: Define depression and facilitation.

Speaker note:
Short-term plasticity has two variables. x_k is the available synaptic resources, which decrease during firing and recover quickly. u_k is the utilization factor, which increases with recent activity and decays slowly. Depression helps terminate bursts; facilitation stores the memory trace.

Transition:
The next slide shows why these two mechanisms work together.

## Slide 15: Why Facilitation and Depression Complement Each Other

Purpose: Explain burst generation and memory storage.

Speaker note:
Depression creates a natural burst cycle. A burst consumes resources, weakened synapses stop the burst, and resources recover. Facilitation lasts longer, so after a burst the synapse remains more effective. Together, depression schedules refresh events and facilitation stores the memory.

Transition:
Now we can write the neural mass model itself.

## Slide 16: 4 - Neural Mass Model

Purpose: Begin the model-equation section.

Speaker note:
This section presents the reduced equations and the multi-population architecture used for working memory.

Transition:
The model has four equations per excitatory population.

## Slide 17: The Four-Equation Model

Purpose: Present the core equations.

Speaker note:
The first two equations describe firing rate and mean voltage. The last two describe synaptic resources and utilization. This is the core achievement: population-level dynamics of a large spiking network are represented by a small set of macroscopic variables.

Transition:
Next, interpret each equation qualitatively.

## Slide 18: Reading the Equations

Purpose: Explain what the variables do.

Speaker note:
r and v are the fast neural variables. They describe activity and population voltage. x and u are slower synaptic variables. x controls depression and resource depletion; u controls facilitation and memory persistence. Working memory emerges from the interaction between these fast and slow variables.

Transition:
The model then extends from one population to multiple item-coding populations.

## Slide 19: Multi-Population Architecture

Purpose: Explain how items are represented.

Speaker note:
Each excitatory population can represent one memory item. All excitatory populations interact with a shared inhibitory population. This shared inhibition is important because it creates competition and timing between different items.

Transition:
The connection structure determines where memory is stored.

## Slide 20: Plastic and Fixed Connections

Purpose: Distinguish plastic excitatory synapses from fixed inhibition.

Speaker note:
Only excitatory-to-excitatory synapses are plastic in the model. The inhibitory couplings are fixed. This makes the assumption very clear: memory is stored in short-term changes in excitatory synapses, while inhibition organizes activity and competition.

Transition:
The next slide focuses on the role of inhibition.

## Slide 21: Role of the Inhibitory Population

Purpose: Explain why inhibition is essential.

Speaker note:
Inhibition prevents runaway excitation and synchrony. It also creates competition between memory items and supports beta-gamma rhythms through a PING-like mechanism. So inhibition is not just stabilizing the system; it is part of the computation.

Transition:
Before using the model, the authors verify it against large simulations.

## Slide 22: 5 - Verification

Purpose: Start validation section.

Speaker note:
This section explains how the reduced model is checked against direct network simulations.

Transition:
There are two comparison models.

## Slide 23: How the Model Is Verified

Purpose: Explain microscopic and mesoscopic validation.

Speaker note:
The authors compare the neural mass model to microscopic STP simulations, where each neuron has its own synaptic variables, and mesoscopic STP simulations, where the synaptic variables are population averages. This gives both a realistic and a computationally aligned validation.

Transition:
Figure 1 shows the key validation result.

## Slide 24: Figure 1 Validation Logic

Purpose: Explain the validation experiment.

Speaker note:
The test uses a single excitatory population and two stimulus pulses. The important observation is that the neural mass model reproduces the population bursts and synaptic variable dynamics. Small differences from microscopic STP come from microscopic correlations not represented in the mesoscopic variables.

Transition:
Now use the figure to make the validation visually clear.

## Slide 25: Key Figure: Model Validation

Purpose: Guide the audience through Fig. 1.

Speaker note:
Point to the overlap between the reduced model and network simulations. Emphasize that the model captures not just steady firing rate, but the transient burst dynamics. This is what justifies using the model for working-memory operations.

Transition:
The validated model is then used to generate different working-memory modes.

## Slide 26: 6 - Working Memory Modes

Purpose: Start the operational modes section.

Speaker note:
The model can reproduce three different working-memory regimes depending mainly on the background current.

Transition:
The next slide previews the three modes.

## Slide 27: Three Modes Controlled by Background Current

Purpose: Preview the three regimes.

Speaker note:
At low background current, memory is silent and retrieved by a cue. At intermediate current, spontaneous bursts refresh the memory. At higher current, persistent activity maintains the item. This is useful because one model unifies three mechanisms often discussed separately.

Transition:
Start with selective reactivation.

## Slide 28: Mode 1: Selective Reactivation

Purpose: Explain silent synaptic memory.

Speaker note:
In this mode, the stimulus increases facilitation in the target population, then activity returns to baseline. The trace is stored in synapses. A weak non-specific cue later reactivates only the facilitated population, so retrieval is selective.

Transition:
The next mode removes the need for an external cue.

## Slide 29: Mode 2: Spontaneous Reactivation

Purpose: Explain burst-based automatic refresh.

Speaker note:
Here the system is in a bistable regime. After loading, the memory population continues producing periodic population bursts. Each burst refreshes facilitation, so the memory can persist without a continuous external signal.

Transition:
The third mode is closer to the classical persistent-activity view.

## Slide 30: Mode 3: Persistent Activity

Purpose: Explain activity-based maintenance.

Speaker note:
At higher background current, the loaded population enters a persistent firing state. The memory is now maintained by ongoing activity, and facilitation remains high. This is more metabolically expensive and behaves differently from silent synaptic storage.

Transition:
Figure 3 compares all three modes visually.

## Slide 31: Key Figure: Three Working-Memory Modes

Purpose: Use the paper result figure.

Speaker note:
Use this figure to show that the same model can move between selective reactivation, spontaneous reactivation, and persistent activity. The key message is unification: changing the dynamical regime changes the memory mechanism.

Transition:
The paper then compares this exact model with heuristic firing-rate models.

## Slide 32: Why the Exact Model Beats the Heuristic Model

Purpose: Explain the limitation of heuristic models.

Speaker note:
Heuristic models can reproduce some broad behavior, but they miss important transient dynamics. In particular, they do not generate the fast beta-gamma oscillations seen during memory loading. This is a qualitative limitation, not just a small numerical error.

Transition:
The figure makes this limitation concrete.

## Slide 33: Key Figure: Heuristic Model Limitation

Purpose: Show why exact dynamics matter.

Speaker note:
Use this slide to emphasize that beta-gamma activity is experimentally observed during working-memory tasks. The exact neural mass model can reproduce those rhythms, while the heuristic model largely cannot.

Transition:
Now move from single-item memory to competition between items.

## Slide 34: 7 - Competition and Juggling

Purpose: Start multi-item competition section.

Speaker note:
This section explains how two memory items compete, switch, or coexist.

Transition:
There are three possible outcomes.

## Slide 35: Two-Item Competition

Purpose: Present the outcome categories.

Speaker note:
When a second item is presented, the first item can win, the second item can win, or both can be maintained. The outcome depends on stimulus strength and duration because those determine how much facilitation each population receives.

Transition:
The coexistence case is called juggling.

## Slide 36: Juggling Mechanism

Purpose: Explain anti-phase bursting.

Speaker note:
In juggling, the two item populations burst in alternation. A burst in one population activates inhibition, which temporarily suppresses the other population. When inhibition relaxes, the other population bursts. This creates separate time slots for each item.

Transition:
The paper visualizes this anti-phase competition.

## Slide 37: Key Figure: Competition and Juggling

Purpose: Show paper figure for item competition.

Speaker note:
If using Fig. 5, point out the anti-phase bursts. If using Fig. 6, point out the outcome map. The main message is that working memory is a dynamical competition, not passive storage in independent slots.

Transition:
Persistent states behave differently.

## Slide 38: Competition in Persistent States

Purpose: Compare burst-based and persistent competition.

Speaker note:
In the burst-based synaptic regime, juggling is possible. In the persistent activity regime, juggling is not observed; the system tends to select one item or the other. This shows that the mechanism of maintenance changes the nature of multi-item competition.

Transition:
Next, the model is extended to several items.

## Slide 39: 8 - Multi-Item Memory

Purpose: Start capacity section.

Speaker note:
Now the model is used with more excitatory populations, each representing a possible item.

Transition:
The first question is how multiple items are loaded.

## Slide 40: Loading Several Items

Purpose: Explain multi-item loading setup.

Speaker note:
The authors use seven excitatory populations and load items sequentially. During each loading event, the stimulated population bursts, while other populations are temporarily suppressed. After loading, the stored populations organize into structured burst timing.

Transition:
That timing structure is called a splay state.

## Slide 41: Splay State Organization

Purpose: Explain phase-separated multi-item storage.

Speaker note:
In a splay state, all stored item populations burst with the same cycle period but different phases. This means each item gets its own time slot. It is a temporal organization of memory items, not a fixed symbolic slot system.

Transition:
This temporal organization creates a natural capacity limit.

## Slide 42: Memory Capacity

Purpose: Present the capacity result.

Speaker note:
The model can maintain up to about five items under optimal conditions. When more items are loaded, some drop out or the system becomes unstable. This is important because the capacity limit emerges from the dynamics rather than being imposed by the model.

Transition:
The next figure shows multi-item loading or dropout.

## Slide 43: Key Figure: Multi-Item Loading and Capacity

Purpose: Use Fig. 8 or Fig. 9.

Speaker note:
If using Fig. 8, emphasize the splay-state timing. If using Fig. 9, emphasize capacity failure and item dropout. The psychological link is that primacy and recency effects can naturally emerge.

Transition:
Capacity also depends on presentation rate.

## Slide 44: Presentation Rate Matters

Purpose: Explain timing dependence.

Speaker note:
Encoding is best when presentation timing matches the natural rhythms of the network. Too slow, and recency dominates. Too fast, and stimuli interfere destructively. The optimal range corresponds to the network's burst dynamics.

Transition:
The paper also gives an analytical capacity estimate.

## Slide 45: Analytical Capacity Formula

Purpose: Explain what controls capacity.

Speaker note:
You do not need to derive this formula in detail. The important point is what it says: capacity increases with depression timescale, excitability, background current, and excitatory coupling, and decreases with strong inhibition. The formula links capacity to biophysical parameters.

Transition:
Compare the formula to simulation.

## Slide 46: Capacity Prediction

Purpose: Show analytical agreement.

Speaker note:
The formula predicts a capacity between about 3.6 and 4.8 items, while simulations show a maximum of 5. This is good agreement and better than previous heuristic estimates.

Transition:
Next, the paper analyzes frequency bands.

## Slide 47: 9 - Frequency-Band Results

Purpose: Start oscillation results.

Speaker note:
The model is not only about item capacity. It also predicts frequency-band signatures comparable with neurophysiological data.

Transition:
Start with spectral signatures during loading and maintenance.

## Slide 48: Spectral Signatures of WM

Purpose: Summarize oscillations.

Speaker note:
During loading, the model produces beta-gamma bursts and low-frequency transients. During maintenance, multi-item burst timing creates structured rhythms. These rhythms come from the interaction between excitatory populations, inhibition, and STP.

Transition:
Then examine how power changes with memory load.

## Slide 49: Power vs Number of Loaded Items

Purpose: Explain load-dependent frequency bands.

Speaker note:
Gamma power increases with memory load, which matches several experiments. Beta is more complex and non-monotonic. Theta is more tied to single excitatory population dynamics. Alpha shows little variation, which is also consistent with some human findings.

Transition:
The figure makes these trends visible.

## Slide 50: Key Figure: Frequency Bands vs Memory Load

Purpose: Show Fig. 11.

Speaker note:
Point out the gamma trend first because it is the clearest. Then mention that beta and theta behave differently, suggesting different mechanisms for different frequency bands.

Transition:
Now connect these patterns to experimental findings.

## Slide 51: Comparison With Experiments

Purpose: Summarize empirical relevance.

Speaker note:
The model reproduces several qualitative findings from human and monkey studies: gamma load effects, beta complexity, weak alpha modulation, beta-gamma loading rhythms, and delta transients. This is why the mean-voltage and oscillation outputs are important.

Transition:
The next section focuses on an ERP-like memory-load measure.

## Slide 52: 10 - Memory Load and ERP Analogy

Purpose: Start voltage-load section.

Speaker note:
This section highlights one of the benefits of tracking mean membrane potential.

Transition:
The authors define a voltage contrast between coding and non-coding populations.

## Slide 53: From Mean Voltage to Memory Load

Purpose: Explain Delta v.

Speaker note:
The authors compute a difference between the mean voltage of populations storing items and populations not storing items. This difference increases with load, saturates near capacity, and decreases when the system is overloaded.

Transition:
This mirrors ERP memory-capacity results.

## Slide 54: Key Figure: ERP-Like Memory Load Signal

Purpose: Show Fig. 12.

Speaker note:
Use this slide to make the experimental bridge explicit. The model produces a voltage-based load signal that behaves like human ERP memory-capacity measures. This result would not be accessible in a firing-rate-only model.

Transition:
Finally, the bifurcation analysis explains why the different modes occur.

## Slide 55: 11 - Bifurcation Picture

Purpose: Start dynamical-systems explanation.

Speaker note:
The bifurcation analysis identifies which stable states exist as background current changes.

Transition:
First, the table summarizes the key transitions.

## Slide 56: Stable States as Background Current Changes

Purpose: Summarize bifurcation points.

Speaker note:
As background current increases, the system moves from a single low-firing state, to bistability, to oscillatory population bursts, and then to persistent activity. These transitions explain why changing I_B changes the working-memory mode.

Transition:
The figure shows the branches directly.

## Slide 57: Key Figure: Bifurcation Diagram

Purpose: Show Fig. 13.

Speaker note:
Point to the regions corresponding to silent storage, burst refresh, and persistent activity. The figure turns the three memory modes into one continuous dynamical story controlled by background current.

Transition:
Use the next slide to connect bifurcation regions to the three modes.

## Slide 58: Bifurcations Explain the Three Modes

Purpose: Tie modes to current values.

Speaker note:
At I_B = 1.2, only low firing is stable, so memory must be silent and synaptic. At I_B = 1.532, self-sustained bursts can refresh memory. At I_B = 2.0, persistent activity is stable. This gives a compact explanation of the whole paper.

Transition:
Now summarize the takeaways.

## Slide 59: 12 - Takeaways

Purpose: Begin final summary.

Speaker note:
This final section collects the model contributions, working-memory contributions, limitations, and future directions.

Transition:
First, the major results.

## Slide 60: Major Results

Purpose: Summarize contributions.

Speaker note:
The model contribution is an exact neural mass reduction with STP and mean voltage. The working-memory contribution is that the same model explains multiple maintenance modes, oscillatory signatures, capacity limits, and ERP-like load signals.

Transition:
The mechanism can be summarized in one loop.

## Slide 61: Core Mechanistic Picture

Purpose: Give the conceptual summary.

Speaker note:
The central loop is: stimulus causes a burst, the burst depletes resources and increases facilitation, depression ends the burst, and facilitation stores the trace. Later, a cue or spontaneous burst refreshes the memory.

Transition:
Now mention what the model leaves out.

## Slide 62: Limitations

Purpose: Acknowledge simplifications.

Speaker note:
The model uses simplified pulsatile interactions, no transmission delays, and a single-layer architecture. It also does not include explicit cognitive control. These are not failures, but boundaries of the current model.

Transition:
Those limitations motivate future work.

## Slide 63: Future Directions

Purpose: Identify next steps.

Speaker note:
Natural extensions include richer synaptic time courses, delayed transmission, second-order synaptic statistics, multi-layer cortical circuits, and more explicit control mechanisms for task demands.

Transition:
Close with the main conclusion.

## Slide 64: Conclusion

Purpose: End with the thesis of the paper.

Speaker note:
The key message is that working memory can be modeled as an exact population-level dynamical system where synaptic facilitation stores information, depression schedules refresh bursts, and inhibition organizes competition and timing. The model bridges spiking networks and measurable population signals like LFP, EEG, and ERP.


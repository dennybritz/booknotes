### Prologue

But learning algorithms are artifacts that design other artifacts.

Amazon’s algorithm, more than any one person, determines what books are read in the world today.

Symbolists view learning as the inverse of deduction and take ideas from philosophy, psychology, and logic. Connectionists reverse engineer the brain and are inspired by neuroscience and physics. Evolutionaries simulate evolution on the computer and draw on genetics and evolutionary biology. Bayesians believe learning is a form of probabilistic inference and have their roots in statistics. Analogizers learn by extrapolating from similarity judgments and are influenced by psychology and mathematical optimization.

On the contrary, what it requires is stepping back from the mathematical arcana to see the overarching pattern of learning phenomena; and for this the layman, approaching the forest from a distance, is in some ways better placed than the specialist, already deeply immersed in the study of particular trees.

### Chapter 1 - The Machine Learning Revolution

(As Richard Feynman said, “What I cannot create, I do not understand.”)

Scientists make theories, and engineers make devices. Computer scientists make algorithms, which are both theories and devices.

Learning algorithms are the seeds, data is the soil, and the learned programs are the grown plants.

The Industrial Revolution automated manual work and the Information Revolution did the same for mental work, but machine learning automates automation itself. Without it, programmers become the bottleneck holding up progress.

In retrospect, we can see that the progression from computers to the Internet to machine learning was inevitable: computers enable the Internet, which creates a flood of data and the problem of limitless choice; and machine learning uses the flood of data to help solve the limitless choice problem.


### Chapter 2 - The Master Algorithm

All knowledge—past, present, and future—can be derived from data by a single, universal learning algorithm.

Thus it seems that evolution kept the cerebellum around not because it does something the cortex can’t, but just because it’s more efficient.

If something exists but the brain can’t learn it, we don’t know it exists. We may just not see it or think it’s random.

But if everything we experience is the product of a few simple laws, then it makes sense that a single algorithm can induce all that can be induced.

Biology, in turn, is the result of optimization by evolution within the constraints of physics and chemistry,

Humans are good at solving NP problems approximately, and conversely, problems that we find interesting (like Tetris) often have an “NP-ness” about them.

In 1962, when Kennedy gave his famous moon-shot speech, going to the moon was an engineering problem. In 1662, it wasn’t, and that’s closer to where AI is today.

To use a technology, we don’t need to master its inner workings, but we do need to have a good conceptual model of it.

The analogizers’ master algorithm is the support vector machine, which figures out which experiences to remember and how to combine them to make new predictions.


### Chapter 3 - Hume's Problem of Induction

The rationalist likes to plan everything in advance before making the first move. The empiricist prefers to try things and see how they turn out.

You could be super-Casanova and have dated millions of women thousands of times each, but your master database still wouldn’t answer the question of what this woman is going to say this time.

How about we just assume that the future will be like the past? This is certainly a risky assumption. (It didn’t work for the inductivist turkey.) On the other hand, without it all knowledge is impossible, and so is life.

result, known as the “no free lunch” theorem, sets a limit on how good a learner can be. The limit is pretty low: no learner can be better than random guessing!

and you have the “no free lunch” theorem. Pick your favorite learner. (We’ll see many in this book.) For every world where it does better than random guessing, I, the devil’s advocate, will deviously construct one where it does worse by the same amount. All I have to do is flip the labels of all unseen instances.

Tom Mitchell, a leading symbolist, calls it “the futility of bias-free learning.” In ordinary life, bias is a pejorative word: preconceived notions are bad. But in machine learning, preconceived notions are indispensable; you can’t learn without them. In fact, preconceived notions are also indispensable to human cognition, but they’re hardwired into the brain, and we take them for granted. It’s biases over and beyond those that are questionable.

Newton’s Principle: Whatever is true of everything we’ve seen is true of everything in the universe.

“All happy families are alike; each unhappy family is unhappy in its own way.” The same is true of individuals. To be happy, you need health, love, friends, money, a job you like, and so on. Take any of these away, and misery ensues.

Learning is forgetting the details as much as it is remembering the important parts.

Consider the little white girl who, upon seeing a Latina baby at the mall, blurted out “Look, Mom, a baby maid!” (True event.)

Bottom line: learning is a race between the amount of data you have and the number of hypotheses you consider. More data exponentially reduces the number of hypotheses that survive, but if you start with a lot of them, you may still have some bad ones left at the end.

Accuracy on held-out data is the gold standard in machine learning.

For example, we can subtract a penalty proportional to the length of the rule from its accuracy and use that as an evaluation measure.

The preference for simpler hypotheses is popularly known as Occam’s razor, but in a machine-learning context this is somewhat misleading. “Entities should not be multiplied beyond necessity,” as the razor is often paraphrased, just means choosing the simplest theory that fits the data.

You can estimate the bias and variance of a learner by comparing its predictions after learning on random variations of the training set. If it keeps making the same mistakes, the problem is bias, and you need a more flexible learner (or just a different one). If there’s no pattern to the mistakes, the problem is variance, and you want to either try a less flexible learner or get more data.

For each pair of facts, we construct the rule that allows us to infer the second fact from the first one and generalize it by Newton’s principle. When the same general rule is induced over and over again, we can have some confidence that it’s true.

This contrasts with traditional chemotherapy, which affects all cells indiscriminately.

Learning which drugs work against which mutations requires a database of patients, their cancers’ genomes, the drugs tried, and the outcomes.

For these, the symbolist algorithm of choice is decision tree induction.

Decision trees instead ensure a priori that each instance will be matched by exactly one rule.

A single concept implicitly defines two classes: the concept itself and its negation. (For example, spam and nonspam.) Classifiers are the most widespread form of machine learning.

So to learn a good decision tree, we pick at each node the attribute that on average yields the lowest class entropy across all its branches, weighted by how many examples go into each branch.

The psychologist David Marr argued that every information processing system should be studied at three distinct levels: the fundamental properties of the problem it’s solving; the algorithms and representations used to solve it; and how they are physically implemented.

Sets of rules and decision trees are easy to understand, so we know what the learner is up to. This makes it easier to figure out what it’s doing right and wrong, fix the latter, and have confidence in the results.

Connectionists, in particular, are highly critical of symbolist learning. According to them, concepts you can define with logical rules are only the tip of the iceberg; there’s a lot going on under the surface that formal reasoning just can’t see, in the same way that most of what goes on in our minds is subconscious.


### Chapter 4 - How Does Your Brain Learn?

“Neurons that fire together wire together.”

brains can perform a large number of computations in parallel, with billions of neurons working at the same time; but each of those computations is slow, because neurons can fire at best a thousand times per second.

Some neurons have short axons and some have exceedingly long ones, reaching clear from one side of the brain to the other. Placed end to end, the axons in your brain would stretch from Earth to the moon.

Perceptrons were invented in the late 1950s by Frank Rosenblatt, a Cornell psychologist.

for example, if one of the memories is the pattern of black-and-white pixels formed by the digit nine and the network sees a distorted nine, it will converge to the “ideal” one and thereby recognize it.

Boltzmann machines could solve the credit-assignment problem in principle, but in practice learning was very slow and painful, making this approach impractical for most applications.

Rather than a logic gate, a neuron is more like a voltage-to-frequency converter. The curve of frequency as a function of voltage looks like this:

When you can’t get the temperature in the shower just right—first it’s too cold, and then it quickly shifts to too hot—blame the S curve.

Many phenomena we think of as linear are in fact S curves, because nothing can grow without limit.

When someone talks about exponential growth, ask yourself: How soon will it turn into an S curve?

Differentiate an S curve and you get a bell curve: slow, fast, slow becomes low, high, low.

This was part of the reason Minsky, Papert, and others couldn’t see how to learn multilayer perceptrons. They could imagine replacing step functions by S curves and doing gradient descent, but then they were faced with the problem of local minima of the error. In those days researchers didn’t trust computer simulations;

Better still, a local minimum may in fact be preferable because it’s less likely to prove to have overfit our data than the global one.

Hyperspace is a double-edged sword. On the one hand, the higher dimensional the space, the more room it has for highly convoluted surfaces and local optima. On the other hand, to be stuck in a local optimum you have to be stuck in every dimension, so it’s more difficult to get stuck in many dimensions than it is in three.

Driverless cars first broke into the public consciousness with the DARPA Grand Challenges in 2004 and 2005, but a over a decade earlier, researchers at Carnegie Mellon had already successfully trained a multilayer perceptron to drive a car by detecting the road in video images and appropriately turning the steering wheel. Carnegie Mellon’s car managed to drive coast to coast across America with very blurry vision (thirty by thirty-two pixels),

Indeed, the history of machine learning itself shows why we need learning algorithms. If algorithms that automatically find related papers in the scientific literature had existed in 1969, they could have potentially helped avoid decades of wasted time and accelerated who knows what discoveries.

The nervous system of the C. elegans worm consists of only 302 neurons and was completely mapped in 1986, but we still have only a fragmentary understanding of what it does.

We don’t build airplanes by reverse engineering feathers, and airplanes don’t flap their wings. Rather, airplane designs are based on the principles of aerodynamics, which all flying objects must obey. We still do not understand those analogous principles of thought.

Neural networks are not compositional, and compositionality is a big part of human cognition. Another big issue is that humans—and symbolic models like sets of rules and decision trees—can explain their reasoning, while neural networks are big piles of numbers that no one can understand.


### Chapter 5 - Evolution: Nature's Learning Algorithm

The key input to a genetic algorithm, as Holland’s creation came to be known, is a fitness function. Given a candidate program and some purpose it is meant to fill, the fitness function assigns the program a numeric score reflecting how well it fits the purpose.

which Holland called classifier systems, are one of the workhorses of the machine-learning tribe he founded: the evolutionaries. Like multilayer perceptrons, classifier systems face the credit-assignment problem—what is the fitness of rules for intermediate concepts?—and Holland devised the so-called bucket brigade algorithm to solve it.

In 1972, Niles Eldredge and Stephen Jay Gould proposed that evolution consists of a series of “punctuated equilibria,” alternating long periods of stasis with short bursts of rapid change, like the Cambrian explosion.

Once the algorithm reaches a local maximum of fitness—a peak in the fitness landscape—it will stay there for a long time until a lucky mutation or crossover lands an individual on the slope to a higher peak, at which point that individual will multiply and climb up the slope with each passing generation. And the higher the current peak, the longer before that happens.

Genetic algorithms, in contrast, are full of random choices: which hypotheses to keep alive and cross over (with fitter hypotheses being more likely candidates), where to cross two strings, which bits to mutate.

Genetic algorithms make no a priori assumptions about the structures they will learn, other than their general form.

Holland showed that, in this case, the fitter a schema’s representatives in one generation are compared to the average, the more of them we can expect to see in the next generation. So, while the genetic algorithm explicitly manipulates strings, it implicitly searches the much larger space of schemas.

A genetic algorithm is like the ringleader of a group of gamblers, playing slot machines in every casino in town at the same time. Two schemas compete with each other if they include the same bits and differ in at least one of them, like *10 and *11, and n competing schemas are like n slot machines. Every set of competing schemas is a casino, and the genetic algorithm simultaneously figures out the winning machine in every casino, following the optimal strategy of playing the better-seeming machines with exponentially increasing frequency. Pretty smart.

One consequence of crossing over program trees instead of bit strings is that the resulting programs can have any size, making the learning more flexible. The overall tendency is for bloat, however, with larger and larger trees growing as evolution goes on longer (also known as “survival of the fattest”).

Genetic programming’s first success, in 1995, was in designing electronic circuits. Starting with a pile of electronic components such as transistors, resistors, and capacitors, Koza’s system reinvented a previously patented design for a low-pass filter, a circuit that can be used for things like enhancing the bass on a dance-music track.

None of Holland’s theoretical results show that crossover actually helps; mutation suffices to exponentially increase the frequency of the fittest schemas in the population over time.

Engineers certainly use building blocks extensively, but combining them involves, well, a lot of engineering; it’s not just a matter of throwing them together any old way, and it’s not clear crossover can do the trick.

No one is sure why sex is pervasive in nature, either.

“It takes all the running you can do, to keep in the same place.” In this view, organisms are in a perpetual arms race with parasites, and sex helps keep the population varied, so that no single germ can infect all of it.

Christos Papadimitriou and colleagues have shown that sex optimizes not fitness but what they call mixability: a gene’s ability to do well on average when combined with other genes. This can be useful when the fitness function is either not known or not constant, as in natural selection, but in machine learning and optimization, hill climbing tends to do better.

With or without crossover, evolving structure is an essential part of the Master Algorithm. The brain can learn anything, but it can’t evolve a brain.

The Master Algorithm is neither genetic programming nor backprop, but it has to include the key elements of both: structure learning and weight learning.

In Baldwinian evolution, behaviors that are first learned later become genetically hardwired. If dog-like mammals can learn to swim, they have a better chance to evolve into seals—as they did—than if they drown.

The architecture of the brain may well have similar faults—the brain has many constraints that computers don’t, like very limited short-term memory—and there’s no reason to stay within them.


### Chapter 6 - In the Church of Reverend Bayes

For Bayesians, learning is “just” another application of Bayes’ theorem, with whole models as the hypotheses and the data as the evidence: as you see more data, some models become more likely and some less, until ideally one model stands out as the clear winner.

From this thought experiment, Laplace derived his so-called rule of succession, which estimates the probability that the sun will rise again after having risen n times as (n + 1) / (n + 2). When n = 0, this is just ½; and as n increases, so does the probability, approaching 1 when n approaches infinity.

P(cause | effect) = P(cause) × P(effect | cause) / P(effect).

Humans, it turns out, are not very good at Bayesian inference, at least when verbal reasoning is involved. The problem is that we tend to neglect the cause’s prior probability.

I put just in quotes because implementing Bayes’ theorem on a computer turns out to be fiendishly hard for all but the simplest problems, for reasons that we’re about to see.

each combination of symptoms and flu/not flu. A learner that uses Bayes’ theorem and assumes the effects are independent given the cause is called a Naïve Bayes classifier.

The economist Milton Friedman even argued in a highly influential essay that the best theories are the most oversimplified, provided their predictions are accurate, because they explain the most with the least.

It might not seem so at first, but Naïve Bayes is closely related to the perceptron algorithm. The perceptron adds weights and Naïve Bayes multiplies probabilities, but if you take a logarithm, the latter reduces to the former. Both can be seen as generalizations of simple If . . . then . . . rules,

If the states and observations are continuous variables instead of discrete ones, the HMM becomes what’s known as a Kalman filter.

A more insidious problem is that with confidence-rated rules we’re prone to double-counting evidence.

everything is connected, but only indirectly. In order to affect me, something that happens a mile away must first affect something in my neighborhood, even if only through the propagation of light. As one wag put it, space is the reason everything doesn’t happen to you. Put another way, the structure of space is an instance of conditional independence.

In retrospect, we can see that Naïve Bayes, Markov chains, and HMMs are all special cases of Bayesian networks. The structure of Naïve Bayes is:

The AIDS virus is a tough adversary because it mutates rapidly, making it difficult for any one vaccine or drug to pin it down for long. Heckerman noticed that this is the same cat-and-mouse game that spam filters play with spam

You could always construct it from the individual tables, but that takes exponential time and space. What we really want is to compute P(Burglary | Bob called, Claire didn’t) without building the full table. That, in a nutshell, is the problem of inference in Bayesian networks.

Burglary and Earthquake are a priori independent, but the alarm going off entangles them: the alarm makes you suspect a burglary, but if now you hear on the radio that there’s been an earthquake, you assume that’s what caused the alarm. The earthquake has explained away the alarm, making a burglary less likely, and the two are therefore dependent.

The trick in MCMC is to design a Markov chain that converges to the distribution of our Bayesian network. One easy option is to repeatedly cycle through the variables, sampling each one according to its conditional probability given the state of its neighbors.

People often talk about MCMC as a kind of simulation, but it’s not: the Markov chain does not simulate any real process; rather, we concocted it to efficiently generate samples from a Bayesian network, which is itself not a sequential model.

This is justified by the so-called maximum likelihood principle: of all the possible probabilities of heads, 0.7 is the one under which seeing seventy heads in a hundred flips is most likely. The likelihood of a hypothesis is P(data | hypothesis), and the principle says we should pick the hypothesis that maximizes it.

For a Bayesian, in fact, there is no such thing as the truth; you have a prior distribution over hypotheses, after seeing the data it becomes the posterior distribution, as given by Bayes’ theorem, and that’s all.

If we’re willing to assume that all hypotheses are equally likely a priori, the Bayesian approach now reduces to the maximum likelihood principle. So Bayesians can say to frequentists: “See, what you do is a special case of what we do, but at least we make our assumptions explicit.”

Bayesians can do something much more interesting. They can use the prior distribution to encode experts’ knowledge about the problem—their answer to Hume’s question. For example, we can design an initial Bayesian network for medical diagnosis by interviewing doctors, asking them which symptoms they think depend on which diseases, and adding the corresponding arrows. This is the “prior network,” and the prior distribution can penalize alternative networks by the number of arrows that they add or remove from it.

We can put a prior distribution on any class of hypotheses—sets of rules, neural networks, programs—and then update it with the hypotheses’ likelihood given the data.

The simplified graph structure makes the models learnable and is worth keeping, but then we’re better off just learning the best parameters we can for the task at hand, irrespective of whether they’re probabilities.

Pandora’s features are handcrafted, but in Markov networks we can also learn features using hill climbing, similar to rule induction. Either way, gradient descent is a good way to learn the weights.

Markov networks can be trained to maximize either the likelihood of the whole data or the conditional likelihood of what we want to predict given what we know. For Siri, the likelihood of the whole data is P(words, sounds), and the conditional likelihood we’re interested in is P(words | sounds). By optimizing the latter, we can ignore P(sounds), which is only a distraction from our goal. And since we ignore it, it can be arbitrarily complex. This is much better than HMMs’ unrealistic assumption that sounds depend solely on the corresponding words, without any influence from the surroundings.

Bayesian learning works on a single table of data, where each column represents a variable (for example, the expression level of one gene) and each row represents an instance (for example, a single microarray experiment, with each gene’s observed expression level). It’s OK if the table has “holes” and measurement errors because we can use probabilistic inference to fill in the holes and average over the errors. But if we have more than one table, Bayesian learning is stuck. It doesn’t know how to, for example, combine gene expression data with data about which DNA segments get translated into proteins, and how in turn the three-dimensional shapes of those proteins cause them to lock on to different parts of the DNA molecule, affecting the expression of other genes. In logic, we can easily write rules relating all of these aspects, and learn them from the relevant combinations of tables—but only provided the tables have no holes or errors.

All of the tribes we’ve met so far have one thing in common: they learn an explicit model of the phenomenon under consideration, whether it’s a set of rules, a multilayer perceptron, a genetic program, or a Bayesian network. When they don’t have enough data to do that, they’re stumped. But analogizers can learn from as little as one example because they never form a model. Let’s see what they do instead.


### Chapter 7: You Are What You Resemble

Analogy was the spark that ignited many of history’s greatest scientific advances. The theory of natural selection was born when Darwin, on reading Malthus’s Essay on Population, was struck by the parallels between the struggle for survival in the economy and in nature.

Nearest-neighbor is the simplest and fastest learning algorithm ever invented. In fact, you could even say it’s the fastest algorithm of any kind that could ever be invented.

class. For instance, we’d like to guess where the border between two countries is, but all we know is their capitals’ locations. Most learners would be stumped, but nearest-neighbor happily guesses that the border is a straight line lying halfway between the two cities:

Scientists routinely use linear regression to predict continuous variables, but most phenomena are not linear. Luckily, they’re locally linear because smooth curves are locally well approximated by straight lines. So if instead of trying to fit a straight line to all the data, you just fit it to the points near the query point, you now have a very powerful nonlinear regression algorithm.

If Kennedy had needed a complete theory of international relations to decide what to do about the Soviet missiles in Cuba, he would have been in trouble. Instead, he saw an analogy between that crisis and the outbreak of World War I, and that analogy guided him to the right decisions.

These days all kinds of algorithms are used to recommend items to users, but weighted k-nearest-neighbor was the first widely used one, and it’s still hard to beat.

So a simple way to make nearest-neighbor more efficient is to delete all the examples that are correctly classified by their neighbors.

Nearest-neighbor was the first algorithm in history that could take advantage of unlimited amounts of data to learn arbitrarily complex concepts.

But nearest-neighbor is hopelessly confused by irrelevant attributes because they all contribute to the similarity between examples. With enough irrelevant attributes, accidental similarity in the irrelevant dimensions swamps out meaningful similarity in the important ones, and nearest-neighbor becomes no better than random guessing.

It gets even worse. Nearest-neighbor is based on finding similar objects, and in high dimensions, the notion of similarity itself breaks down. Hyperspace is like the Twilight Zone. The intuitions we have from living in three dimensions no longer apply, and weird and weirder things start to happen. Consider an orange: a tasty ball of pulp surrounded by a thin shell of skin. Let’s say 90 percent of the radius of an orange is occupied by pulp, and the remaining 10 percent by skin. That means 73 percent of the volume of the orange is pulp (0.93). Now consider a hyperorange: still with 90 percent of the radius occupied by pulp, but in a hundred dimensions, say. The pulp has shrunk to only about three thousandths of a percent of the hyperorange’s volume (0.9100). The hyperorange is all skin, and you’ll never be done peeling it!

With a high-dimensional normal distribution, you’re more likely to get a sample far from the mean than close to it. A bell curve in hyperspace looks more like a doughnut than a bell.

In fact, no learner is immune to the curse of dimensionality. It’s the second worst problem in machine learning, after overfitting. The term curse of dimensionality was coined by Richard Bellman, a control theorist, in the fifties.

To handle weakly relevant attributes, one option is to learn attribute weights. Instead of letting the similarity along all dimensions count equally, we “shrink” the less-relevant ones.

This “blessing of nonuniformity,” whereby data is not spread uniformly in (hyper) space, is often what saves the day. The examples may have a thousand attributes, but in reality they all “live” in a much lower-dimensional space.

each pixel is a dimension, so there are many, but only a tiny fraction of all possible images are digits, and they all live together in a cozy little corner of hyperspace.

the SVM chooses the support vectors and weights that yield the maximum possible margin.

we have to maximize the margin under the constraint that the weights can only increase up to some fixed value. Or, equivalently, we can minimize the weights under the constraint that all examples have a given margin, which could be one—the precise value is arbitrary. This is what SVMs usually do.

SVMs can be seen as a generalization of the perceptron, because a hyperplane boundary between classes is what you get when you use a particular similarity measure (the dot product between vectors). But SVMs have a major advantage compared to multilayer perceptrons: the weights have a single optimum instead of many local ones and so learning them reliably is much easier.

Despite this, SVMs are no less expressive than multilayer perceptrons; the support vectors effectively act as a

hidden layer and their weighted average as the output layer.

Provided you can learn them, networks with many layers can express many functions more compactly than SVMs, which always have just one layer, and this can make all the difference.

It turns out that we can view what SVMs do with kernels, support vectors, and weights as mapping the data to a higher-dimensional space and finding a maximum-margin hyperplane in that space. For some kernels, the derived space has infinite dimensions, but SVMs are completely unfazed by that. Hyperspace may be the Twilight Zone, but SVMs have figured out how to navigate it.

If Cope is right, creativity—the ultimate unfathomable—boils down to analogy and recombination. Judge for yourself by googling “david cope mp3.”

Structure mapping takes two descriptions, finds a coherent correspondence between some of their parts and relations, and then, based on that correspondence, transfers further properties from one structure to the other. For example, if the structures are the solar system and the atom, we can map planets to electrons and the sun to the nucleus and conclude, as Bohr did, that electrons revolve around the nucleus.

When little Tim sees women looking after other children like his mother looks after him, he generalizes the concept “mommy” to mean anyone’s mommy, not just his. That in turn is a springboard for understanding things like “mother ship” and “Mother Nature.”

Rules are in effect generalized instances where we’ve “forgotten” some attributes because they didn’t matter.

As we go through life, similar episodes gradually become abstracted into rule-based structures, like “eating at a restaurant.”

The problem is that all the learners we’ve seen so far need a teacher to tell them the right answer. They can’t learn to distinguish tumor cells from healthy ones unless someone labels them “tumor” or “healthy.” But humans can learn without a teacher; they do it from the day they’re born.


### Chapter 8 - Learnning Without a Teacher

If we could revisit ourselves as infants and toddlers and see the world again through those newborn eyes, much of what puzzles us about learning—even about existence itself—would suddenly seem obvious.

Above all, even though children certainly get plenty of help from their parents, they learn mostly on their own, without supervision, and that’s what seems most miraculous.

A young baby is not surprised if a teddy bear passes behind a screen and reemerges as an airplane, but a one-year-old is.

Whenever we want to learn a statistical model but are missing some crucial information (e.g., the classes of the examples), we can use EM.

You might have noticed a certain resemblance between k-means and EM, in that they both alternate between assigning entities to clusters and updating the clusters’ descriptions. This is not an accident: k-means itself is a special case of EM, which you get when all the attributes have “narrow” normal distributions, that is, normal distributions with very small variance.

The famous hockey-stick curve of global warming, for example, is the result of finding the principal component of various temperature-related data series (tree rings, ice cores, etc.) and assuming it’s the temperature.

One of the most popular algorithms for nonlinear dimensionality reduction, called Isomap, does just this. It connects each data point in a high-dimensional space (a face, say) to all nearby points (very similar faces), computes the shortest distances between all pairs of points along the resulting network and finds the reduced coordinates that best approximate these distances.

Here’s an interesting experiment. Take the video stream from Robby’s eyes, treat each frame as a point in the space of images, and reduce that set of images to a single dimension. What will you discover? Time. Like a librarian arranging books on a shelf, time places each image next to its most similar ones. Perhaps our perception of it is just a natural result of our brains’ dimensionality reduction prowess. In the road network of memory, time is the main thoroughfare, and we soon find it. Time, in other words, is the principal component of memory.

Humans do have one constant guide: their emotions. We seek pleasure and avoid pain.

Pleasure travels back through time, so to speak, and actions can eventually become associated with effects that are quite remote from them.

Children’s play is a lot more serious than it looks; if evolution made a creature that is helpless and a heavy burden on its parents for the first several years of its life, that extravagant cost must be for the sake of an even bigger benefit. In effect, reinforcement learning is a kind of speeded-up evolution—trying, discarding, and refining actions within a single lifetime instead of over generations—and by that standard it’s extremely efficient.

Gaming aside, researchers have used reinforcement learning to balance poles, control stick-figure gymnasts, park cars backward, fly helicopters upside down, manage automated telephone dialogues, assign channels in cell phone networks, dispatch elevators, schedule space-shuttle cargo loading, and much else.

Chris Watkins, on the other hand, is dissatisfied. He sees many things children can do that reinforcement learners can’t: solve problems, solve them better after a few attempts, make plans, acquire increasingly abstract knowledge. Luckily, we also have learning algorithms for these higher-level abilities, the most important of which is chunking.

Pretty much every human skill follows a power law, with different powers for different skills.

Crucially, grouping things into chunks allows us to process much more information than we otherwise could. That’s why telephone numbers have hyphens: 1-723-458-3897 is much easier to remember than 17234583897.

A chunk in this sense has two parts: the stimulus (a pattern you recognize in the external world or in your short-term memory) and the response (the sequence of actions you execute as a result).

In nonrelational learning, the parameters of a model are tied in only one way: across all the independent examples (e.g., all the patients we’ve diagnosed). In relational learning, every feature template we create ties the parameters of all its instances.

According to Seldon, people are like molecules in a gas, and the law of large numbers ensures that even if individuals are unpredictable, whole societies aren’t. Relational learning reveals why this is not the

case. If people were independent, each making decisions in isolation, societies would indeed be predictable, because all those random decisions would add up to a fairly constant average. But when people interact, larger assemblies can be less predictable than smaller ones, not more.


### Chapter 9 - The Pieces Of The Puzzle Fall Into Place

Although it is less well known, many of the most important technologies in the world are the result of inventing a unifier, a single mechanism that does what previously required many.

As it turns out, it’s not hard to combine many different learners into one, using what is known as metalearning. Netflix, Watson, Kinect, and countless others use it, and it’s one of the most powerful arrows in the machine learner’s quiver. It’s also a stepping-stone to the deeper unification that will follow.

Bagging generates random variations of the training set by resampling, applies the same learner to each one, and combines the results by voting.

One of the cleverest metalearners is boosting, created by two learning theorists, Yoav Freund and Rob Schapire. Instead of combining different learners, boosting repeatedly applies the same classifier to the data, using each new model to correct the previous ones’ mistakes. It does this by assigning weights to the training examples; the weight of each misclassified example is increased after each round of learning, causing later rounds to focus more on it.

As you approach it from a distance, you can see that the city is made up of three concentric circles, each bounded by a wall. The outer and by far widest circle is Optimization Town. Each house here is an algorithm, and they come in all shapes and sizes. Some are under construction, the locals busy around them; some are gleaming new; and some look old and abandoned. Higher up the hill lies the Citadel of Evaluation. From its mansions and palaces orders issue continuously to the algorithms below. Above all, silhouetted against the sky, rise the Towers of Representation.



Representation is the formal language in which the learner expresses its models. The symbolists’ formal language is logic, of which rules and decision trees are special cases. The connectionists’ is neural networks. The evolutionaries’ is genetic programs, including classifier systems. The Bayesians’ is graphical models, an umbrella term for Bayesian networks and Markov networks. The analogizers’ is specific instances, possibly with weights, as in an SVM.

The evaluation component is a scoring function that says how good a model is. Symbolists use accuracy or information gain. Connectionists use a continuous error measure, such as squared error, which is the sum of the squares of the differences between the predicted values and the true ones. Bayesians use the posterior probability. Analogizers (at least of the SVM stripe) use the margin. In addition to how well the model fits the data, all tribes take into account other desirable properties, such as the model’s simplicity.

Optimization is the algorithm that searches for the highest-scoring

model and returns it. The symbolists’ characteristic search algorithm is inverse deduction. The connectionists’ is gradient descent. The evolutionaries’ is genetic search, including crossover and mutation. The Bayesians are unusual in this regard: they don’t just look for the best model, but average over all models, weighted by how probable they are. To do the weighting efficiently, they use probabilistic inference algorithms like MCMC. The analogizers (or more precisely, the SVM mavens) use constrained optimization to find the best model.

This is what nature does: evolution creates brain structures, and individual experience modulates them.

It looks like you’ve boiled down the five optimizers to a simple recipe: genetic search for structure and gradient descent for parameters. And even that may be overkill. For a lot of problems, you can whittle genetic search down to hill climbing if you do three things: leave out crossover, try all possible point mutations in each generation, and always select the single best hypothesis to seed the next generation.

You use accuracy to evaluate yes-or-no predictions and squared error for continuous ones. Fitness is just the evolutionaries’ name for the scoring function; you can make it anything you want, including accuracy and squared error. Posterior probability reduces to squared error if you ignore the prior probability and the errors follow a normal distribution. The margin, if you allow it to be violated for a price, becomes a softer version of accuracy: instead of paying no penalty for a correct prediction and a penalty of one for an incorrect prediction, the penalty is zero until you get inside the margin, at which point it starts to steadily go up.

Suddenly you see it: an SVM is just a multilayer perceptron with a hidden layer composed of kernels instead of S curves and an output that’s a linear combination instead of another S curve.

Each rule is just a highly stylized neuron. For example, the rule If it’s a giant reptile and breathes fire then it’s a dragon is just a perceptron with weights of one for it’s a giant reptile and breathes fire and a threshold of 1.5. And a set of rules is a multilayer perceptron with a hidden layer containing one neuron for each rule and an output neuron to form the disjunction of the rules.

product of factors is now a sum of terms, just like an SVM, a voting set of rules, or a multilayer perceptron without the output S curve.

Everyone has the flu and If someone has the flu, so do their friends. In standard logic, this would be a pretty useless pair of statements: the first would rule out any state with even a single healthy person, and the second would be redundant. But in an MLN, the first formula just means that there’s a feature X has the flu for every person X, with the same weight as the formula. If people are likely to have the flu, the formula will have a high weight, and so will the corresponding features. A state with many healthy people is less probable than one with few, but not impossible. And because of the second formula, a state where someone has the flu and their friends don’t is less probable than one where healthy and infected people fall into separate clusters of friends.

On the one hand, assuming the learner is part of the world is an assumption—in principle, the learner could obey different laws from those the world obeys—so it satisfies Hume’s dictum that learning is only possible with prior knowledge. On the other hand, it’s an assumption so basic and hard to disagree with that perhaps it’s all we need for this world.

Backpropagation is a form of gradient descent,

The world is not a random jumble of interactions; it has a hierarchical structure: galaxies, planets, continents, countries, cities, neighborhoods, your house, you, your head, your nose, a cell on its tip, the organelles in it, molecules, atoms, subatomic particles. The way to model it, then, is with an MLN that also has a hierarchical structure.

If you look at it one way, machine learning is only a small part of the CanceRx project, well behind data gathering and human contributions.


### Chapter 10 - This Is The World Of Machine Learning

The novelty in the world today is that computers, not just people, are starting to have theories of mind.

If you don’t like a company, click on their ads: this will not only waste their money now, but teach Google to waste it again in the future by showing the ads to people who are unlikely to buy the products.

In this rapidly approaching future, you’re not going to be the only one with a “digital half” doing your bidding twenty-four hours a day. Everyone will have a detailed model of him- or herself, and these models will talk to each other all the time.

but what about a model of what makes a hotel good or bad for you? This requires information about you that you may not want to share with TripAdvisor. What you’d like is a trusted party that combines the two types of data and gives you the results.

These problems all have a common solution: a new type of company that is to your data what your bank is to your money.

Common sense is important not just because your mom taught you so, but because computers don’t have it.

ATMs replaced some bank tellers, but mainly they let us withdraw money any time, anywhere.

We worry that the humanities are in a death spiral, but they’ll rise from the ashes once other professions have been automated. The more everything is done cheaply by machines, the more valuable the humanist’s contribution will be.

Eventually, we’ll start talking about the employment rate instead of the unemployment one and reducing it will be seen as a sign of progress. (“The US is falling behind. Our employment rate is still 23 percent.”)

People will seek meaning in human relationships, self-actualization, and spirituality, much as they do now. The need to earn a living will be a distant memory, another piece of humanity’s barbaric past that we rose above.

Technology is the extended phenotype of man. This means we can continue to control it even if it becomes far more complex than we can understand.

People worry that computers will get too smart and take over the world, but the real problem is that they’re too stupid and they’ve already taken over the world.

As a matter of fact, we’ve always lived in a world that we only partly understood. The main difference is that our world is now partly created by us, which is surely an improvement. The world beyond the Turing point will not be incomprehensible to us, any more than the Pleistocene was. We’ll focus on what we can understand, as we always have, and call the rest random (or divine).

The statistician knows that prediction is hard, especially about the future, and the computer scientist knows that the best way to predict the future is to invent it, but the unexamined future is not worth inventing.
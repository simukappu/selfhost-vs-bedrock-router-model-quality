# How this evaluation dataset was built

The 200 scenarios in `scenarios.json` are the easy part to look at and the hard part to produce. The models, the prompt, and the grading rule are all downstream of one question that no benchmark harness answers for you: for a given user utterance, what is the *correct* thing for the agent to do? This note is about how we answered that question, because the answer turned out to be more reusable than the numbers it produced.

## The first approach, and why it stalled

The obvious way to build a test set is to enumerate. List the query characteristics (search hints present or absent, phrasing, complexity), cross them with the UI context the request starts from (top page, search results, product page), cross that with conversation state (nothing yet, a product already shown, an order already discussed), and you get a grid of patterns to cover. We started there. On paper the three axes multiply out to around ninety pattern combinations before conversation history is even added.

That grid never closes. For any cell you can find one more phrasing, one more page context, one more prior turn that changes the right answer, and the enumeration keeps branching. We were writing down combinations faster than we were deciding what the agent should actually do in them, and the backlog of "patterns to still cover" grew every time we looked at it. A purely logical enumeration of surface patterns does not terminate, and chasing coverage that way would have taken effectively forever.

## The pivot: decide the behavior first, per use case

What broke the stall was inverting the order. Instead of enumerating inputs and then labeling them, we fixed a small set of use cases (seven intents), and for each one we wrote down the assumptions that decide the correct behavior. Those assumptions, not the scenario count, are the artifact worth keeping. Once they exist, generating scenarios is mechanical, and two people label the same utterance the same way.

The assumptions are business and UX rulings, and most of them are not obvious from the words alone:

- **Returns and exchanges are always a policy question**, no matter how the user phrases it. "I want to return this" reads like an action, but a return is a physical process the agent cannot complete through an API, so it routes to FAQ, not to an order-action tool. Cancellations and address changes *are* API-completable, so those can be actions. The dividing line is what the backend can actually do, not the grammar of the request.
- **"Can I cancel?" and "Cancel it" are different intents.** The first asks whether something is possible and is answered as policy. The second expresses intent to act. Same verb, different job.
- **General subject versus personal subject** splits two intents that look alike. "How long does delivery usually take?" is a general policy answer. "When will *my* order arrive?" reads the user's own data. "What is the point expiry rule?" is policy; "What is my point balance?" is a data lookup.
- **The page context can flip the intent for an identical utterance.** "Do you have size M?" is a stock check on a product page, a search refinement on a results page, and a request that needs clarification from the top page. A greeting, by contrast, stays a greeting on every page, because the utterance carries no product or search intent to be reinterpreted.
- **Some rulings are pure product decisions.** This service has no coupon feature, so every coupon utterance ("show my coupons", "can I stack coupons?") is answered as a policy FAQ explaining that coupons are not offered, and there is no coupon tool at all. Nothing in the query tells you that. It comes from knowing what the product does and does not do.

Multi-step tool flows fall out of UX assumptions the same way. A user does not remember order IDs, so "cancel my order" cannot go straight to a cancel call. It has to list the user's orders first, resolve which one, then act. The dataset encodes that two-step expectation because that is the behavior the experience requires, not because the sentence implied two calls.

## Why you cannot hand this to a model

A language model is good at producing query variations and even at guessing plausible labels. It cannot make the rulings above, because they are decisions about what the *right* behavior is, and that depends on facts the model does not have: whether returns are API-completable in your system, whether coupons exist, whether a capability question should be treated as policy or as an action in your product. Generate labels from a model without fixing these first and you get a dataset that is internally consistent and wrong in the ways that matter, because it has quietly invented a product that is not yours.

Deciding the behavior is closer to product work than to data work. It is a review of the use cases one at a time, asking what the user is trying to do and what a correct response looks like in the context they are in. The value of this dataset is the set of encoded rulings; the scenarios are just those rulings made concrete enough to grade against.

## What ended up in the dataset

Each scenario carries the user input, the page context, an expected intent (one of seven), and an expected tool set, plus an allowance for tools that are acceptable but not required. Grading is a set match on tool names and an exact match on intent; argument values are not scored here. The full contract (the seven intents, the thirteen tool names, the JSON schema) lives alongside this file in `schema.json`, and the per-scenario expectations are in `scenarios.json`.

The search tool accepts a small semantic-tag layer in addition to exact filters: capability (waterproof, lightweight), occasion, season, target audience, and style, separate from structured fields like color, size, and price. Splitting a query into "exact filter" and "semantic tag" is itself one of the encoded assumptions, and it shapes what a correct `product_search` call looks like.

## The takeaway worth carrying

If you are standing up agent evaluation, budget most of the effort for defining correct behavior, not for collecting inputs. The inputs are cheap and a model can help with them. The definitions are where the work is, they do not generalize across products, and they are the part you will reuse every time the model, the prompt, or the tool set changes underneath you.

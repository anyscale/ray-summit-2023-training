# Real-time ML - Deploying Online Inference Pipelines

© 2023, Anyscale Inc. All Rights Reserved

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/Ray-Thumbnail-Adam.png" width="71%" loading="lazy">

<a href="https://forms.gle/9TSdDYUgxYs8SA9e8"><img src="https://img.shields.io/badge/Ray-Join%20Slack-blue" alt="join-ray-slack"></a>
<a href="https://discuss.ray.io/"><img src="https://img.shields.io/badge/Discuss-Ask%20Questions-blue" alt="discuss"></a>
<a href="https://twitter.com/raydistributed"><img src="https://img.shields.io/twitter/follow/raydistributed?label=Follow" alt="twitter"></a>

## Overview

Once our AI/ML models are ready for deployment, that's when the fun really starts. We need our AI-powered services to be resilient and efficient, scalable to demand and adaptable to heterogeneous environments (like using GPUs or TPUs as effectively as possible). Moreover, when we build applications around online inference, we often need to integrate different services: multiple models, data sources, business logic, and more.

Ray Serve was built so that we can easily overcome all of those challenges.

In this class we'll learn to use Ray Serve to compose online inference applications meeting all of these requirements and more. We'll build services that integrate with each other while autoscaling individually, even supporting individual hardware and software requirements -- all using regular Python and often with just one new line of code.

## Motivating Scenario: Multilingual LLM Chat

For our example use case, we’ll see how to leverage Ray Serve to host a LLM Chat model and how to enhance it using additional services for multilingual interactions.

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/system-overview-multilingual-chat.jpg" width="60%" loading="lazy">


## Learning Outcomes

* Develop a deep understanding of the various architectural components of Ray Serve.
* Use deployments and deployment graphs API to serve machine learning models in production environments for online inference.
* Combine multiple models to build complex logic, allowing for a more sophisticated machine learning pipeline.

## Topics discussed

* Context of Ray Serve
* Deployments
* Service resources (e.g., CPU/GPU/...)
* Runtime environments and dependencies
* Imperative pattern for service composition
* Declarative (graph) pattern for service composition
* Architecture / Under-the-hood
* Scaling and Performance
* Request batching
* Additional production tools, tips, and patterns

## Connect with the Ray community

You can learn and get more involved with the Ray community of developers and researchers:

* [**Ray documentation**](https://docs.ray.io/en/latest)

* [**Official Ray site**](https://www.ray.io/)
Browse the ecosystem and use this site as a hub to get the information that you need to get going and building with Ray.

* [**Join the community on Slack**](https://forms.gle/9TSdDYUgxYs8SA9e8)
Find friends to discuss your new learnings in our Slack space.

* [**Use the discussion board**](https://discuss.ray.io/)
Ask questions, follow topics, and view announcements on this community forum.

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Generic/ray_logo.png" width="30%" loading="lazy">

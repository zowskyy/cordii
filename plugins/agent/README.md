# agent — routing, loop, and orchestration

Agent-facing plugins that own routing, parameter extraction, multi-domain dispatch, aggregation, and the main event loop.

## Plugins

- AgentLoop — main turn loop
- MultiDomainRouter — domain selection
- ParameterExtractor — argument normalization
- QuerySplitter — decomposition
- AggregateResponse — response merging
- SemanticRouter — optional embedding-based routing (full profile only)

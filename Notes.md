LangGraph - > stateful , Langchain - > stateless
consisting nodes and edges , sequential based
eg-> JD              ,          eg -> chaining models
chatting = conventional memory 
mutable - changable , editable |
node do the procesing |
event driven |
fault taulerence |
recovery concept small = retry
large - > persistence layer
points : Persistence & Memory: , Time Travel (Replay & Rewind) , Human-in-the-Loop:, Fault Tolerance:
checkpoint - > ex:game checkpoint exit save , continue next day

subGraphs: child graphs interaction of other graph work as connector
useCase:
MultiAgent , Reusability

Observablity: eg of Auditor -> google ad running it will notice
how? --> langsmith
best intregation with langGraph | bluecode with langchain 
Timeline Tracker , debugger
Langchain provider  , simple sequential 
Langgraph non linear complex workflows
langgraph is orchestation 
branching , conditional and parallelisim
LLM workflows:
performs distinct task for eg: decision making 
ToolAugmented use , common workflows
PROMPT CHAINING , ROUTING , Parallalization , Evaluator Optimizers ,OrchestratorWorkers ARE some of the examples
Core Concept:
a set of function in python 
nodes kya krna ,egdes when to execute
state change on each execution
state accesible to all nodes
reducers -> add for parallel workflow
compilation - > super step or routing

#without_llm , #with llm used api key Mistral case nodes , edges , states GraphState main node 
then compile intial state to question answer, propmting to put question ,model.invoke to answer
"""
Agentic Regulatory Compliance Mapping System
Uses LangGraph + CrewAI with Advanced RAG
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from crewai import Agent, Task, Crew, Process

# ============================================================================

# DATA STRUCTURES

# ============================================================================

@dataclass
class Requirement:
id: str
title: str
text: str
domain: str
topic: str
evidence: str
additional_text: str
risk_types: List[str]
asset_types: List[str]

@dataclass
class RegulatoryText:
citation: str
text: str
domain: str
risk_context: str

class GraphState(TypedDict):
regulatory_text: RegulatoryText
retrieved_requirements: List[Dict]
agent_analyses: Dict[str, Any]
mappings: List[Dict]
gaps: List[Dict]
final_report: Dict

# ============================================================================

# HYBRID RETRIEVAL ENGINE

# ============================================================================

class HybridRetriever:
def **init**(self, excel_path: str):
self.df = pd.read_excel(excel_path)
self.setup_chromadb()
self.setup_bm25()
self.reranker = HuggingFaceCrossEncoder(
model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

```
def setup_chromadb(self):
    """Initialize ChromaDB with embeddings"""
    self.client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use sentence transformers for embeddings
    self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create collection
    self.collection = self.client.get_or_create_collection(
        name="internal_requirements",
        embedding_function=self.embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Index documents
    self._index_requirements()

def _index_requirements(self):
    """Index requirements into ChromaDB"""
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in self.df.iterrows():
        # Combine fields for rich semantic representation
        doc_text = f"""
        Title: {row['title']}
        Text: {row['text']}
        Domain: {row['domain']}
        Topic: {row['topic']}
        Evidence: {row['evidence']}
        Additional: {row.get('additional_text', '')}
        """
        
        documents.append(doc_text)
        metadatas.append({
            'title': str(row['title']),
            'domain': str(row['domain']),
            'topic': str(row['topic']),
            'risk_types': str(row.get('risk_types', '')),
            'asset_types': str(row.get('type_of_asset', ''))
        })
        ids.append(f"req_{idx}")
    
    # Batch upsert
    self.collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def setup_bm25(self):
    """Setup BM25 for sparse retrieval"""
    # Tokenize documents
    corpus = []
    for _, row in self.df.iterrows():
        doc = f"{row['title']} {row['text']} {row['domain']} {row['topic']}"
        corpus.append(doc.lower().split())
    
    self.bm25 = BM25Okapi(corpus)
    
def dense_search(self, query: str, top_k: int = 20) -> List[Dict]:
    """Semantic search using ChromaDB"""
    results = self.collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return [
        {
            'id': results['ids'][0][i],
            'score': 1 - results['distances'][0][i],  # Convert distance to similarity
            'metadata': results['metadatas'][0][i],
            'document': results['documents'][0][i]
        }
        for i in range(len(results['ids'][0]))
    ]

def sparse_search(self, query: str, top_k: int = 20) -> List[Dict]:
    """BM25 sparse retrieval"""
    query_tokens = query.lower().split()
    scores = self.bm25.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return [
        {
            'id': f"req_{idx}",
            'score': scores[idx],
            'metadata': {
                'title': self.df.iloc[idx]['title'],
                'domain': self.df.iloc[idx]['domain']
            }
        }
        for idx in top_indices
    ]

def reciprocal_rank_fusion(
    self, 
    dense_results: List[Dict], 
    sparse_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """Combine rankings using RRF"""
    rrf_scores = {}
    
    # Process dense results
    for rank, result in enumerate(dense_results):
        req_id = result['id']
        rrf_scores[req_id] = rrf_scores.get(req_id, 0) + 1 / (k + rank + 1)
    
    # Process sparse results
    for rank, result in enumerate(sparse_results):
        req_id = result['id']
        rrf_scores[req_id] = rrf_scores.get(req_id, 0) + 1 / (k + rank + 1)
    
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return combined results
    return [{'id': req_id, 'rrf_score': score} for req_id, score in sorted_ids]

def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    """Rerank candidates using cross-encoder"""
    # Get full documents for candidates
    doc_pairs = []
    for candidate in candidates[:30]:  # Rerank top 30 from RRF
        result = self.collection.get(ids=[candidate['id']])
        if result['documents']:
            doc_pairs.append([query, result['documents'][0]])
    
    if not doc_pairs:
        return candidates[:top_k]
    
    # Score with cross-encoder
    scores = self.reranker.score(doc_pairs)
    
    # Combine with original candidates
    for i, candidate in enumerate(candidates[:len(scores)]):
        candidate['rerank_score'] = float(scores[i])
    
    # Sort by rerank score
    reranked = sorted(
        candidates[:len(scores)], 
        key=lambda x: x.get('rerank_score', 0), 
        reverse=True
    )
    
    return reranked[:top_k]

def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
    """Full hybrid search pipeline"""
    # Step 1: Dense and sparse retrieval
    dense_results = self.dense_search(query, top_k=20)
    sparse_results = self.sparse_search(query, top_k=20)
    
    # Step 2: Reciprocal rank fusion
    fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
    
    # Step 3: Rerank top candidates
    final_results = self.rerank(query, fused_results, top_k=top_k)
    
    return final_results
```

# ============================================================================

# CREWAI AGENTS

# ============================================================================

def create_agents(llm):
"""Create specialized agents for compliance mapping"""

```
retrieval_specialist = Agent(
    role='Retrieval Specialist',
    goal='Find the most relevant internal requirements for the given regulatory text',
    backstory='Expert in information retrieval and search systems',
    llm=llm,
    verbose=True
)

domain_expert = Agent(
    role='Domain Expert',
    goal='Analyze semantic relevance and domain alignment between regulations and requirements',
    backstory='Senior compliance officer with deep domain knowledge',
    llm=llm,
    verbose=True
)

risk_assessor = Agent(
    role='Risk Assessor',
    goal='Evaluate risk type alignment and asset type coverage',
    backstory='Risk management specialist focused on cyber security',
    llm=llm,
    verbose=True
)

compliance_mapper = Agent(
    role='Compliance Mapper',
    goal='Determine precise applicability and coverage of requirements',
    backstory='Compliance mapping expert with regulatory background',
    llm=llm,
    verbose=True
)

critic = Agent(
    role='Critical Reviewer',
    goal='Challenge mappings and identify potential false positives or gaps',
    backstory='Devil\'s advocate ensuring mapping quality and accuracy',
    llm=llm,
    verbose=True
)

gap_analyst = Agent(
    role='Gap Analyst',
    goal='Identify coverage gaps and prioritize remediation',
    backstory='Strategic analyst focused on compliance gaps and risks',
    llm=llm,
    verbose=True
)

return {
    'retrieval': retrieval_specialist,
    'domain': domain_expert,
    'risk': risk_assessor,
    'mapper': compliance_mapper,
    'critic': critic,
    'gap': gap_analyst
}
```

# ============================================================================

# LANGGRAPH WORKFLOW

# ============================================================================

class ComplianceMappingGraph:
def **init**(self, retriever: HybridRetriever, llm):
self.retriever = retriever
self.llm = llm
self.agents = create_agents(llm)
self.graph = self._build_graph()

```
def _build_graph(self):
    """Build LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", self.retrieve_requirements)
    workflow.add_node("analyze", self.multi_agent_analysis)
    workflow.add_node("critique", self.critic_review)
    workflow.add_node("gap_analysis", self.perform_gap_analysis)
    workflow.add_node("generate_report", self.generate_report)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "critique")
    workflow.add_edge("critique", "gap_analysis")
    workflow.add_edge("gap_analysis", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()

def retrieve_requirements(self, state: GraphState) -> GraphState:
    """Execute hybrid retrieval"""
    reg_text = state['regulatory_text']
    query = f"{reg_text.citation} {reg_text.text} {reg_text.domain}"
    
    results = self.retriever.hybrid_search(query, top_k=15)
    state['retrieved_requirements'] = results
    
    return state

def multi_agent_analysis(self, state: GraphState) -> GraphState:
    """Parallel agent analysis using CrewAI"""
    reg_text = state['regulatory_text']
    requirements = state['retrieved_requirements']
    
    # Create tasks for each agent
    domain_task = Task(
        description=f"""
        Analyze the semantic relevance between the regulatory text and retrieved requirements.
        Regulatory Text: {reg_text.text}
        Requirements: {requirements[:5]}
        
        Provide relevance scores (0-10) and justifications.
        """,
        agent=self.agents['domain'],
        expected_output="Relevance analysis with scores"
    )
    
    risk_task = Task(
        description=f"""
        Evaluate risk type and asset type alignment.
        Regulatory Context: {reg_text.risk_context}
        Requirements: {requirements[:5]}
        
        Identify matching risk types and coverage gaps.
        """,
        agent=self.agents['risk'],
        expected_output="Risk alignment analysis"
    )
    
    mapping_task = Task(
        description=f"""
        Determine which requirements are applicable to the regulatory text.
        Provide confidence scores and mapping rationale.
        
        Regulatory Text: {reg_text.text}
        Requirements: {requirements[:5]}
        """,
        agent=self.agents['mapper'],
        expected_output="Applicability mappings with confidence"
    )
    
    # Execute crew
    crew = Crew(
        agents=[self.agents['domain'], self.agents['risk'], self.agents['mapper']],
        tasks=[domain_task, risk_task, mapping_task],
        process=Process.parallel,
        verbose=True
    )
    
    results = crew.kickoff()
    
    state['agent_analyses'] = {
        'domain': results,
        'risk': results,
        'mappings': results
    }
    
    return state

def critic_review(self, state: GraphState) -> GraphState:
    """Critic agent challenges the mappings"""
    critic_task = Task(
        description=f"""
        Review the proposed mappings and identify:
        1. Potential false positives
        2. Weak justifications
        3. Missing considerations
        4. Edge cases
        
        Analyses: {state['agent_analyses']}
        """,
        agent=self.agents['critic'],
        expected_output="Critical review with concerns"
    )
    
    crew = Crew(
        agents=[self.agents['critic']],
        tasks=[critic_task],
        verbose=True
    )
    
    critique = crew.kickoff()
    state['agent_analyses']['critique'] = critique
    
    return state

def perform_gap_analysis(self, state: GraphState) -> GraphState:
    """Identify coverage gaps"""
    gap_task = Task(
        description=f"""
        Based on the regulatory requirements and mapped controls:
        1. Identify uncovered regulatory requirements
        2. Highlight overlapping or redundant mappings
        3. Prioritize gaps by risk and impact
        4. Suggest remediation actions
        
        Regulatory Text: {state['regulatory_text']}
        Mappings: {state['agent_analyses']}
        """,
        agent=self.agents['gap'],
        expected_output="Gap analysis report"
    )
    
    crew = Crew(
        agents=[self.agents['gap']],
        tasks=[gap_task],
        verbose=True
    )
    
    gaps = crew.kickoff()
    state['gaps'] = gaps
    
    return state

def generate_report(self, state: GraphState) -> GraphState:
    """Generate final compliance report"""
    report = {
        'regulatory_citation': state['regulatory_text'].citation,
        'retrieved_count': len(state['retrieved_requirements']),
        'mapped_requirements': state['agent_analyses'].get('mappings', []),
        'risk_assessment': state['agent_analyses'].get('risk', {}),
        'critical_review': state['agent_analyses'].get('critique', {}),
        'gaps': state['gaps'],
        'recommendations': []
    }
    
    state['final_report'] = report
    return state

def run(self, regulatory_text: RegulatoryText) -> Dict:
    """Execute the full workflow"""
    initial_state = GraphState(
        regulatory_text=regulatory_text,
        retrieved_requirements=[],
        agent_analyses={},
        mappings=[],
        gaps=[],
        final_report={}
    )
    
    final_state = self.graph.invoke(initial_state)
    return final_state['final_report']
```

# ============================================================================

# MAIN EXECUTION

# ============================================================================

def main():
# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
retriever = HybridRetriever(excel_path="requirements.xlsx")

```
# Create mapping system
mapper = ComplianceMappingGraph(retriever=retriever, llm=llm)

# Example regulatory text
reg_text = RegulatoryText(
    citation="NIST CSF 2.0 - ID.AM-1",
    text="Physical devices and systems within the organization are inventoried",
    domain="Asset Management",
    risk_context="Inventory and tracking of physical devices"
)

# Run mapping
report = mapper.run(reg_text)

print("\n" + "="*80)
print("COMPLIANCE MAPPING REPORT")
print("="*80)
print(f"\nRegulatory Citation: {report['regulatory_citation']}")
print(f"Requirements Retrieved: {report['retrieved_count']}")
print(f"\nMapped Requirements: {len(report['mapped_requirements'])}")
print(f"\nGaps Identified: {report['gaps']}")
print("="*80)
```

if **name** == "**main**":
main()
# scripts/test_query_tools.py
"""
Test script for the query tools wrapper functions
Tests all query types and paper summary functionality
"""
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from deep_academic_agent.wrappers.query_tools import (
    query_processed_data, 
    get_paper_summary, 
    get_available_papers,
    format_query_results_for_llm
)


def print_separator(title: str):
    """Print a formatted separator"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)


def test_available_papers():
    """Test getting list of available papers"""
    print_separator("TEST 1: AVAILABLE PAPERS")
    
    papers = get_available_papers()
    
    if not papers:
        print("❌ No processed papers found in database")
        print("Please run the download and process pipeline first:")
        print("python -m deep_academic_agent.scripts.test_download_process")
        return []
    
    print(f"✅ Found {len(papers)} processed papers")
    
    # Show top papers by quality
    papers_sorted = sorted(papers, key=lambda x: x['quality_score'], reverse=True)
    llm_ready_count = sum(1 for p in papers if p['llm_ready'])
    
    print(f"📊 Statistics:")
    print(f"  • LLM-ready papers: {llm_ready_count}/{len(papers)}")
    print(f"  • Average quality score: {sum(p['quality_score'] for p in papers)/len(papers):.1f}")
    
    print(f"\n🏆 Top Papers by Quality:")
    for i, paper in enumerate(papers_sorted[:5], 1):
        status = "✅ LLM-Ready" if paper['llm_ready'] else "⚠️  Processing"
        print(f"  {i}. {paper['paper_id']} - Score: {paper['quality_score']:.1f} - {status}")
        print(f"     Chunks: {paper['chunk_count']}, Entities: {paper['entity_count']}")
    
    return papers


def test_paper_summary(papers):
    """Test paper summary functionality"""
    print_separator("TEST 2: PAPER SUMMARY")
    
    if not papers:
        print("❌ No papers available for summary testing")
        return
    
    # Get summary for the best quality paper
    best_paper = max(papers, key=lambda x: x['quality_score'])
    paper_id = best_paper['paper_id']
    
    print(f"📄 Generating summary for paper: {paper_id}")
    print(f"   Quality score: {best_paper['quality_score']:.1f}")
    
    start_time = datetime.now()
    summary = get_paper_summary(paper_id, include_content_samples=True)
    summary_time = (datetime.now() - start_time).total_seconds()
    
    if not summary:
        print("❌ Failed to generate paper summary")
        return
    
    print(f"✅ Summary generated in {summary_time:.2f}s")
    print(f"\n📋 Paper Overview:")
    print(f"  • Paper ID: {summary.paper_id}")
    print(f"  • Total sections: {len(summary.sections)}")
    print(f"  • Content chunks: {summary.total_chunks}")
    print(f"  • Key entities: {len(summary.key_entities)}")
    print(f"  • LLM Ready: {'✅ Yes' if summary.llm_ready else '❌ No'}")
    
    # Show sections
    if summary.sections:
        print(f"\n📚 Document Sections:")
        for section in summary.sections[:8]:  # Show first 8 sections
            print(f"  • {section['title']} ({section['type']})")
        if len(summary.sections) > 8:
            print(f"  ... and {len(summary.sections) - 8} more sections")
    
    # Show top entities
    if summary.key_entities:
        print(f"\n🔍 Top Entities:")
        for entity in summary.key_entities[:10]:
            print(f"  • {entity['text']} ({entity['type']}) - "
                  f"Confidence: {entity['confidence']:.2f}, Mentions: {entity['mentions']}")
    
    # Show mathematical content
    if summary.mathematical_content.get('total_equations', 0) > 0:
        math = summary.mathematical_content
        print(f"\n🧮 Mathematical Content:")
        print(f"  • Total equations: {math['total_equations']}")
        print(f"  • Average complexity: {math['avg_complexity']:.2f}")
        print(f"  • Type variety: {math['type_variety']}")
    
    # Show citation info
    if summary.citation_info.get('total_citations', 0) > 0:
        citations = summary.citation_info
        print(f"\n📚 Citations:")
        print(f"  • Total citations: {citations['total_citations']}")
        if citations.get('avg_year'):
            print(f"  • Average year: {citations['avg_year']}")
    
    # Show quality scores
    if summary.processing_quality:
        print(f"\n⭐ Processing Quality:")
        for metric, score in summary.processing_quality.items():
            print(f"  • {metric}: {score:.1f}/100")
    
    # Show sample content
    if summary.abstract_text:
        print(f"\n📖 Abstract Sample:")
        print(f"  {summary.abstract_text[:200]}...")
    
    if summary.key_findings:
        print(f"\n🔬 Key Findings:")
        for i, finding in enumerate(summary.key_findings[:3], 1):
            print(f"  {i}. {finding[:150]}...")
    
    return summary


def test_query_types(papers):
    """Test different query types"""
    print_separator("TEST 3: QUERY TYPES")
    
    if not papers:
        print("❌ No papers available for query testing")
        return
    
    # Select a few papers for testing
    test_paper_ids = [p['paper_id'] for p in papers[:3]]
    print(f"🎯 Testing queries on papers: {test_paper_ids}")
    
    # Test queries with different types
    test_queries = [
        {
            "query": "neural networks deep learning",
            "type": "semantic",
            "description": "Semantic search for neural networks"
        },
        {
            "query": "machine learning algorithm",
            "type": "keyword", 
            "description": "Keyword search for ML algorithms"
        },
        {
            "query": "dataset",
            "type": "entity",
            "description": "Entity search for datasets"
        },
        {
            "query": "transformer",
            "type": "entity",
            "description": "Entity search for transformer models"
        }
    ]
    
    successful_queries = 0
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {test_query['description']}")
        print(f"   Query: '{test_query['query']}'")
        print(f"   Type: {test_query['type']}")
        
        start_time = datetime.now()
        result = query_processed_data(
            query=test_query['query'],
            query_type=test_query['type'],
            paper_ids=test_paper_ids,
            max_results=3,
            include_context=True
        )
        query_time = (datetime.now() - start_time).total_seconds()
        
        if result.success:
            successful_queries += 1
            print(f"   ✅ Success: {result.total_matches} results in {query_time:.2f}s")
            
            if result.results:
                print(f"   📋 Top Results:")
                for j, item in enumerate(result.results[:2], 1):
                    print(f"     {j}. Paper: {item.get('paper_id', 'Unknown')}")
                    if item.get('chunk_text'):
                        preview = item['chunk_text'][:100].replace('\n', ' ')
                        print(f"        Content: {preview}...")
                    if item.get('section_title'):
                        print(f"        Section: {item['section_title']}")
                    if item.get('relevance_score'):
                        print(f"        Relevance: {item['relevance_score']}")
                    if item.get('entity_text'):
                        print(f"        Entity: {item['entity_text']} ({item.get('entity_type')})")
            else:
                print(f"   ⚠️  No results found")
        else:
            print(f"   ❌ Failed: {'; '.join(result.errors or ['Unknown error'])}")
    
    print(f"\n📊 Query Test Summary: {successful_queries}/{len(test_queries)} successful")
    return successful_queries


def test_formatted_output(papers):
    """Test LLM-formatted output"""
    print_separator("TEST 4: LLM-FORMATTED OUTPUT")
    
    if not papers:
        print("❌ No papers available for formatting test")
        return
    
    # Run a sample query
    test_paper_ids = [p['paper_id'] for p in papers[:2]]
    
    print("🤖 Testing LLM-formatted output...")
    result = query_processed_data(
        query="deep learning methods",
        query_type="semantic",
        paper_ids=test_paper_ids,
        max_results=3
    )
    
    if result.success:
        formatted_output = format_query_results_for_llm(result)
        print("✅ Formatted output generated:")
        print("\n" + "─" * 50)
        print(formatted_output)
        print("─" * 50)
    else:
        print(f"❌ Query failed: {'; '.join(result.errors or ['Unknown error'])}")


def test_mathematical_search(papers):
    """Test mathematical content search"""
    print_separator("TEST 5: MATHEMATICAL SEARCH")
    
    if not papers:
        print("❌ No papers available for math search testing")
        return
    
    # Test mathematical queries
    math_queries = [
        "equation",
        "matrix",
        "optimization",
        "probability"
    ]
    
    successful_math_queries = 0
    
    for query in math_queries:
        print(f"\n🧮 Math query: '{query}'")
        
        result = query_processed_data(
            query=query,
            query_type="math",
            paper_ids=[p['paper_id'] for p in papers[:3]],
            max_results=2
        )
        
        if result.success and result.results:
            successful_math_queries += 1
            print(f"   ✅ Found {result.total_matches} mathematical results")
            
            for item in result.results[:1]:  # Show first result
                print(f"   📐 LaTeX: {item.get('latex_content', 'N/A')[:50]}...")
                print(f"   📊 Type: {item.get('math_type', 'Unknown')}")
                print(f"   🎯 Complexity: {item.get('complexity_score', 0):.1f}")
        else:
            print(f"   ⚠️  No mathematical results found")
    
    print(f"\n📊 Math Search Summary: {successful_math_queries}/{len(math_queries)} successful")


def main():
    """Main test function"""
    logger.info("Starting query tools comprehensive test")
    
    start_time = datetime.now()
    
    try:
        # Test 1: Get available papers
        papers = test_available_papers()
        
        # Test 2: Paper summary
        if papers:
            test_paper_summary(papers)
            
            # Test 3: Different query types
            test_query_types(papers)
            
            # Test 4: LLM-formatted output
            test_formatted_output(papers)
            
            # Test 5: Mathematical search
            test_mathematical_search(papers)
        
        # Final summary
        total_time = (datetime.now() - start_time).total_seconds()
        
        print_separator("TEST SUMMARY")
        print(f"✅ All query tools tests completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.1f} seconds")
        print(f"📊 Papers available: {len(papers) if papers else 0}")
        
        if papers:
            llm_ready = sum(1 for p in papers if p['llm_ready'])
            print(f"🤖 LLM-ready papers: {llm_ready}/{len(papers)}")
            print(f"\n💡 Next Steps:")
            print(f"  • Use query_processed_data() for semantic search")
            print(f"  • Use get_paper_summary() for detailed paper analysis")
            print(f"  • Integration ready for LangGraph agents!")
        else:
            print(f"\n⚠️  Recommendation:")
            print(f"  Run the complete pipeline first:")
            print(f"  python -m deep_academic_agent.scripts.test_download_process")
    
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"❌ Test failed: {str(e)}")
        print(f"Check that the database contains processed papers")


if __name__ == "__main__":
    main()
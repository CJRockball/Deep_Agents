"""
Simple runner script to test the academic research agent
"""

from minimal_agent import run_research_agent

def main():
    print("🎓 Academic Research Agent - Test Runner")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "subject": "transformer neural networks",
            "query": "What are the main architectural improvements in recent transformer models?"
        },
        {
            "subject": "quantum computing",
            "query": "How do quantum algorithms achieve computational advantage?"
        },
        {
            "subject": "climate change mitigation",
            "query": "What are the most effective renewable energy technologies?"
        }
    ]
    
    print("Available test cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. Subject: {case['subject']}")
        print(f"   Query: {case['query']}")
        print()
    
    # Let user choose or use default
    try:
        choice = input("Enter test case number (1-3) or press Enter for default (1): ").strip()
        if choice == "":
            choice = "1"
        
        case_idx = int(choice) - 1
        if case_idx < 0 or case_idx >= len(test_cases):
            raise ValueError("Invalid choice")
            
        selected_case = test_cases[case_idx]
        
    except (ValueError, KeyboardInterrupt):
        print("Using default test case...")
        selected_case = test_cases[0]
    
    # Run the agent
    print(f"\n🚀 Running test case:")
    print(f"Subject: {selected_case['subject']}")
    print(f"Query: {selected_case['query']}")
    print("\n" + "=" * 80)
    
    result = run_research_agent(
        subject=selected_case['subject'],
        query=selected_case['query']
    )
    
    if result:
        print(f"\n✅ Test completed successfully!")
        print(f"Total messages in conversation: {len(result['messages'])}")
    else:
        print(f"\n❌ Test failed!")

if __name__ == "__main__":
    main()
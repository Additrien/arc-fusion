#!/bin/bash

# Test script for follow-up questions functionality

echo "🧪 Testing Follow-up Questions Implementation"
echo "============================================="

# Generate a unique session ID for this test
SESSION_ID="test-session-$(date +%s)"
echo "Using session ID: $SESSION_ID"

# First question
echo ""
echo "📝 First Question:"
echo "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?"

FIRST_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo ""
echo "✅ First Response:"
echo "$FIRST_RESPONSE" | jq '.answer' -r
echo ""
echo "Session ID: $(echo "$FIRST_RESPONSE" | jq '.session_id' -r)"
echo "Agent Path: $(echo "$FIRST_RESPONSE" | jq '.agent_path' -r)"
echo "Confidence: $(echo "$FIRST_RESPONSE" | jq '.confidence')"

# Wait a moment between requests
sleep 2

# Follow-up question
echo ""
echo "📝 Follow-up Question:"
echo "What accuracy did it achieve?"

SECOND_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What accuracy did it achieve?\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo ""
echo "✅ Follow-up Response:"
echo "$SECOND_RESPONSE" | jq '.answer' -r
echo ""
echo "Session ID: $(echo "$SECOND_RESPONSE" | jq '.session_id' -r)"
echo "Agent Path: $(echo "$SECOND_RESPONSE" | jq '.agent_path' -r)"
echo "Confidence: $(echo "$SECOND_RESPONSE" | jq '.confidence')"

# Test another follow-up
echo ""
echo "📝 Second Follow-up Question:"
echo "How does that compare to other methods mentioned in the paper?"

THIRD_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"How does that compare to other methods mentioned in the paper?\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo ""
echo "✅ Second Follow-up Response:"
echo "$THIRD_RESPONSE" | jq '.answer' -r
echo ""
echo "Session ID: $(echo "$THIRD_RESPONSE" | jq '.session_id' -r)"
echo "Agent Path: $(echo "$THIRD_RESPONSE" | jq '.agent_path' -r)"
echo "Confidence: $(echo "$THIRD_RESPONSE" | jq '.confidence')"

echo ""
echo "🔍 Test Complete!"
echo "Check if the follow-up questions understood the context from previous answers."

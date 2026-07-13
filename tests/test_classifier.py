from lazy_agent_router.classifiers.keyword import KeywordIntentClassifier


def test_keyword_classifier_prefers_more_specific_match():
    classifier = KeywordIntentClassifier({"query": ["查询"], "approve": ["审批", "批准"]})
    assert classifier.predict("查询审批并批准")["intent"] == "approve"

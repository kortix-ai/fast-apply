instruction_format = """Apply changes from <update_snippet> to <original_code>. Output only the complete updated code in <full_updated_code> tag."""


context = """<original_code>{original_code}</original_code>
<update_snippet>{update_snippet}</update_snippet>"""
response_format="""<full_updated_code>{final_code}</full_updated_code>"""

category="closed_qa"


# Example of jsonl
#  {"instruction": "When did Virgin Australia start operating?", "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route. It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.", "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.", "category": "closed_qa"}
#  {"instruction": "Which is a species of fish? Tope or Rope", "context": "", "response": "Tope", "category": "classification"}
#  {"instruction": "Why can camels survive for long without water?", "context": "", "response": "Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.", "category": "open_qa"}

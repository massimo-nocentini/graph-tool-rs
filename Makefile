# Rustdoc for the whole workspace, published under docs/ (e.g. for GitHub
# Pages, "deploy from branch, /docs folder"). docs/ is generated output: the
# prose lives in `//!` comments, starting at crates/gt-core/src/design.rs.

CARGO    ?= cargo
DOC_OUT  := target/doc
DOCS_DIR := docs
# The page docs/index.html redirects to.
DOC_HOME := gt_core/index.html

.PHONY: doc doc-clean

doc:
	rm -rf $(DOC_OUT)
	$(CARGO) doc --workspace --no-deps
	rm -rf $(DOCS_DIR)
	cp -r $(DOC_OUT) $(DOCS_DIR)
	rm -f $(DOCS_DIR)/.lock
	printf '<!DOCTYPE html>\n<meta charset="utf-8">\n<meta http-equiv="refresh" content="0; url=%s">\n<link rel="canonical" href="%s">\n<a href="%s">graph-tool-rs documentation</a>\n' \
		$(DOC_HOME) $(DOC_HOME) $(DOC_HOME) > $(DOCS_DIR)/index.html
	touch $(DOCS_DIR)/.nojekyll

doc-clean:
	rm -rf $(DOCS_DIR)

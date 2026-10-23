from __future__ import annotations

from rlocr.cross_validation import grouped_folds


def test_grouped_folds_have_no_overlap_and_cover_each_test_page_once() -> None:
    pages = [(f"corpus_a/page_{index:03d}", "sidebar" if index % 2 else "table") for index in range(37)]
    pages += [(f"corpus_b/page_{index:03d}", "sidebar") for index in range(23)]
    folds = grouped_folds(pages, folds=5, validation_fraction=0.15, seed=17)
    tested: list[str] = []
    all_pages = {key for key, _ in pages}
    for fold in folds:
        train, validation, test = set(fold.train), set(fold.validation), set(fold.test)
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert train | validation | test == all_pages
        tested.extend(fold.test)
    assert len(tested) == len(all_pages)
    assert set(tested) == all_pages
    assert grouped_folds(pages, seed=17) == folds

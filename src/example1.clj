(ns example1
  (:require [scicloj.ml.core :as ml]
            [scicloj.ml.dataset :as ds]
            [scicloj.ml.metamorph :as mm]
            [tech.v3.datatype.functional :as fun]
            [tech.v3.datatype :as dtype]))


(def dataset1
  (ds/dataset [[:clojure 10 1 0 0]
               [:clojure 10 2 0 1]
               [:clojure 100 20 0 0]
               [:clojure 10 1 1 1]
               [:clojure 10 1 1 0]
               [:clojure 100 5 20 0]
               [:haskell 10 0 2 0]
               [:haskell 100 5 10 0]
               [:haskell 10 1 3 0]
               [:haskell 10 0 4 0]
               [:haskell 10 1 6 0]
               [:haskell 10 0 1 0]]
              {:column-names
               [:topic :words :clojure :haskell :java]}))

(def ds-split
  (-> dataset1
      (ds/split->seq :holdout
                     {:ratio [0.7 0.3]
                      :split-names [:train :test]
                      :seed 3})
      first))

(defn normalize-counts [dataset]
  (-> dataset
      (ds/add-column :words #(-> % :words dtype/->double-array))
      (ds/add-column :clojure #(fun// (:clojure %)
                                      (:words %)))
      (ds/add-column :haskell #(fun// (:haskell %)
                                      (:words %)))
      (ds/add-column :java #(fun// (:java %)
                                   (:words %)))))

(-> ds-split
    :train
    normalize-counts)

(def pipeline1
  (ml/pipeline
   (mm/categorical->number [:topic])
   (mm/set-inference-target :topic)
   (ml/lift normalize-counts)
   {:metamorph/id :model}
   (mm/model {:model-type :smile.classification/logistic-regression})))


(def trained-ctx
  (-> {:metamorph/data (:train ds-split)
       :metamorph/mode :fit}
      pipeline1))

(def predicted-ctx
  (-> trained-ctx
      (merge {:metamorph/data (:test ds-split)
              :metamorph/mode :transform})
      pipeline1))

(-> trained-ctx
    :metamorph/data)

(-> trained-ctx
    :metamorph/data
    :topic
    meta)

(-> predicted-ctx
    :metamorph/data)

(-> predicted-ctx
    :model
    ml/thaw-model
    (.coefficients))

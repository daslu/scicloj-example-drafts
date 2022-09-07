(ns example0
  (:require [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            ;; dtype-next
            [tech.v3.datatype.functional :as fun]
            [tech.v3.datatype :as dtype]
            ;; fastmath
            [fastmath.random :as random]
            ;; clay
            [scicloj.clay.v1.api :as clay]
            [scicloj.clay.v1.tools :as tools]
            [scicloj.clay.v1.extensions :as extensions]
            [scicloj.kindly.v2.kind :as kind]
            [scicloj.viz.api :as viz]))

;; scicloj

;; scicloj.ml:   machine learning (wrapping a few libraries)
;;   dtype-next: processing arrays
;;   tech.ml.dataset: processing tables
;;   tablecloth: nice ergonomics around that
;;   fastmath: math

;; (Deep Diamond (Dragan Djuric): deep learning)


;; For data visualizations, you can use [Clay](https://scicloj.github.io/clay/)
(clay/start! {:tools [tools/scittle]
              :extensions [extensions/dataset]})

;; logistic regression

(defn sigmoid [xs]
  (-> xs
      fun/-
      fun/exp
      (fun/+ 1)
      fun//))

;; sigmoid(x) = 1/(exp(-x)+1) for all x

(sigmoid [-9 -1 0 1 9])

(sigmoid [9])

(def dataset1
  (let [rng (random/rng :isaac 1)
        n 1000
        xs (repeatedly n #(random/grandom rng))
        ws (-> xs
               (fun/* -3)
               (fun/+ 2)
               sigmoid)
        ys (->> ws
                (map #(random/brandom rng %)))]
    (ds/dataset {:x xs :w ws :y ys})))

;; :clojure 44 :haskell 30
;; 2*44 - 1*30 ----sigmoid-->  probability of a clojure blog post

(* 44 -3)

(random/brand 0.5) ; fair coin
(random/brand 0.84) ; unfair coin
(random/brand 0.999) ; unfair coin


(-> dataset1
    viz/data
    (viz/type :point)
    (viz/viz :X :x
             :Y :w))

(def ds-split
  (-> dataset1
      (ds/select-columns [:x :y])
      (ds/split->seq :holdout
                     {:ratio [0.7 0.3]
                      :split-names [:train :test]
                      :seed 3})
      first))

(def pipeline1
  (ml/pipeline
   (mm/categorical->number [:y])
   (mm/set-inference-target :y)
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
    :y
    meta)

(-> predicted-ctx
    :metamorph/data)

(-> predicted-ctx
    :model
    ml/thaw-model
    (.coefficients)
    seq)

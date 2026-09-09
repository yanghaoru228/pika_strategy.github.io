---
layout: default
title: About
---
<div class="tabs">
  <input type="radio" name="tabs" id="tabone" checked="checked">
  <label for="tabone">Tab One</label>
  <div class="tab">
    <p>Introduction: discuss what the project is about.</p>
  </div>
  
  <input type="radio" name="tabs" id="tabtwo">
  <label for="tabtwo">Tab Two</label>
  <div class="tab">
    <p>Data: source of data, EDA, preliminary observations.</p>
  </div>

  <input type="radio" name="tabs" id="tabthree">
  <label for="tabthree">Tab Two</label>
  <div class="tab">
    <p>Analysis: use model.</p>
  </div>

  <input type="radio" name="tabs" id="tabfour">
  <label for="tabfour">Tab Two</label>
  <div class="tab">
    <p>Conclusion: answer, limit, next step.</p>
  </div>
</div>

<style>
.tabs { display: flex; flex-wrap: wrap; }
.tabs label { order: 1; display: block; padding: 10px 20px; margin-right: 4px; cursor: pointer; background: #e0e0e0; font-weight: bold; }
.tabs .tab { order: 99; flex-grow: 1; width: 100%; display: none; padding: 20px; background: #fff; border: 1px solid #ccc; }
.tabs input[type="radio"] { display: none; }
.tabs input[type="radio"]:checked + label { background: #fff; border: 1px solid #ccc; border-bottom: 1px solid transparent; }
.tabs input[type="radio"]:checked + label + .tab { display: block; }
</style>
